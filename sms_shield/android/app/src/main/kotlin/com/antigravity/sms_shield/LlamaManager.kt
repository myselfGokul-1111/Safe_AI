package com.antigravity.sms_shield

import android.content.Context
import android.os.Handler
import android.os.Looper
import android.util.Log
import org.pytorch.executorch.extension.llm.LlmCallback
import org.pytorch.executorch.extension.llm.LlmGenerationConfig
import org.pytorch.executorch.extension.llm.LlmModule
import org.pytorch.executorch.extension.llm.LlmModuleConfig
import java.io.File
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import java.util.concurrent.RejectedExecutionException
import java.util.concurrent.locks.ReentrantLock
import kotlin.concurrent.withLock

/**
 * Standalone native manager for Meta Llama 3.2 1B Instruct inference via ExecuTorch 1.6.0.
 *
 * Responsibilities:
 * - Validates offline model and tokenizer files in app-specific external storage.
 * - Manages the lifecycle of the native [LlmModule] with memory-mapped loading (LOAD_MODE_MMAP).
 * - Executes loading and synchronous generation exclusively on a dedicated background thread.
 * - Queues all concurrent loadModel() listeners and dispatches results to all listeners safely.
 * - Guarantees safe native resource cleanup if loading fails partway.
 * - Provides genuinely idempotent close() with queueing of close callbacks.
 * - Recovers safely from RejectedExecutionException on executor shutdown races.
 * - Isolates callback exceptions so one listener's crash cannot affect others.
 * - Manages process-scoped singleton lifecycle ensuring closed instances are never returned.
 * - Strictly acts as an explanation assistant without altering classification authority.
 */
class LlamaManager private constructor(private val appContext: Context) {

    companion object {
        private const val TAG = "LlamaManager"
        const val MODEL_FILENAME = "llama1b_q4.pte"
        const val TOKENIZER_FILENAME = "tokenizer.model"
        // Must match the max_seq_len / max_context_len used when the .pte was exported (512).
        const val MODEL_CONTEXT_TOKENS = 512
        // Prompt tokens + new tokens must fit inside MODEL_CONTEXT_TOKENS.
        private const val DEFAULT_MAX_NEW_TOKENS = 96
        private const val DEFAULT_TEMPERATURE = 0.1f
        // Max SMS characters placed in the prompt (keeps the prompt well under the context limit).
        private const val MAX_SMS_CHARS = 300

        /**
         * Builds a Llama 3.x Instruct chat prompt that asks for a short plain-language explanation.
         * [verdict] is the classifier's decision (e.g. "SCAM" or "SAFE"); the model only explains it.
         */
        fun buildExplanationPrompt(smsText: String, verdict: String): String {
            val sms = smsText.trim().replace(Regex("\\s+"), " ").take(MAX_SMS_CHARS)
            return "<|begin_of_text|>" +
                "<|start_header_id|>system<|end_header_id|>\n\n" +
                "You explain SMS scam checks to ordinary phone users. " +
                "The verdict is already decided; never change it. " +
                "Reply in 2 short sentences of plain English, naming the specific warning signs in the message." +
                "<|eot_id|>" +
                "<|start_header_id|>user<|end_header_id|>\n\n" +
                "Verdict: $verdict\nSMS: \"$sms\"\n\nExplain why this verdict was given." +
                "<|eot_id|>" +
                "<|start_header_id|>assistant<|end_header_id|>\n\n"
        }

        @Volatile
        private var instance: LlamaManager? = null

        /**
         * Returns the process-scoped LlamaManager singleton.
         * Ensures that if the previous instance was closed, a fresh active instance is created.
         * Does not tie lifecycle to temporary Activity instances, surviving Activity recreation.
         */
        fun getInstance(context: Context): LlamaManager {
            val current = instance
            if (current != null && current.currentState != State.CLOSED && !current.isClosing) {
                return current
            }
            return synchronized(this) {
                val existing = instance
                if (existing != null && existing.currentState != State.CLOSED && !existing.isClosing) {
                    existing
                } else {
                    LlamaManager(context.applicationContext).also { instance = it }
                }
            }
        }

        /**
         * Explicitly closes and resets the singleton instance.
         */
        fun resetInstance(onClosed: (() -> Unit)? = null) {
            val oldInstance = synchronized(this) {
                val inst = instance
                instance = null
                inst
            }
            if (oldInstance != null) {
                oldInstance.close(onClosed)
            } else {
                onClosed?.invoke()
            }
        }
    }

    enum class State {
        UNINITIALIZED,
        LOADING,
        READY,
        GENERATING,
        ERROR,
        CLOSED
    }

    interface GenerationCallback {
        fun onSuccess(response: String)
        fun onError(error: String)
    }

    interface LoadCallback {
        fun onLoaded()
        fun onError(error: String)
    }

    private val mainHandler = Handler(Looper.getMainLooper())
    private val lifecycleLock = ReentrantLock()

    @Volatile
    private var currentState: State = State.UNINITIALIZED

    @Volatile
    private var isClosing: Boolean = false

    @Volatile
    private var lastErrorMessage: String? = null

    private var llmModule: LlmModule? = null

    // Accumulates all registered callbacks while the model is loading
    private val pendingLoadCallbacks = mutableListOf<LoadCallback>()

    // Accumulates all registered close callbacks for idempotent completion
    private val pendingCloseCallbacks = mutableListOf<() -> Unit>()

    private val executor: ExecutorService = Executors.newSingleThreadExecutor { runnable ->
        Thread(runnable, "llama-executorch-worker").apply {
            priority = Thread.NORM_PRIORITY
        }
    }

    /**
     * Directory holding model artifacts: context.getExternalFilesDir("models")
     */
    fun getModelsDirectory(): File? {
        return appContext.getExternalFilesDir("models")
    }

    fun getModelFile(): File? {
        val dir = getModelsDirectory() ?: return null
        return File(dir, MODEL_FILENAME)
    }

    fun getTokenizerFile(): File? {
        val dir = getModelsDirectory() ?: return null
        return File(dir, TOKENIZER_FILENAME)
    }

    /**
     * Checks whether both model and tokenizer exist and have non-zero size.
     */
    fun areModelFilesPresent(): Boolean {
        val model = getModelFile() ?: return false
        val tokenizer = getTokenizerFile() ?: return false
        return model.exists() && model.length() > 0 && tokenizer.exists() && tokenizer.length() > 0
    }

    /**
     * Checks if the native model is fully loaded and ready to accept inference requests.
     */
    fun isModelReady(): Boolean {
        return currentState == State.READY && llmModule != null && !isClosing
    }

    /**
     * Returns the current lifecycle state of the manager.
     */
    fun getState(): State = currentState

    /**
     * Returns the last recorded error message, if any.
     */
    fun getLastError(): String? = lastErrorMessage

    /**
     * Loads the model and tokenizer asynchronously on the dedicated background executor.
     * All registered callbacks (including those registered while loading is in progress)
     * are guaranteed to receive the eventual completion or error notification.
     */
    fun loadModel(callback: LoadCallback? = null) {
        val shouldSubmit: Boolean
        lifecycleLock.withLock {
            if (currentState == State.READY) {
                mainHandler.post {
                    try {
                        callback?.onLoaded()
                    } catch (t: Throwable) {
                        Log.e(TAG, "Uncaught exception in LoadCallback.onLoaded", t)
                    }
                }
                return
            }
            if (currentState == State.LOADING) {
                Log.d(TAG, "Model is already loading; queuing callback.")
                callback?.let { pendingLoadCallbacks.add(it) }
                return
            }
            if (currentState == State.CLOSED || isClosing) {
                val err = "Cannot load model: LlamaManager is closed."
                mainHandler.post {
                    try {
                        callback?.onError(err)
                    } catch (t: Throwable) {
                        Log.e(TAG, "Uncaught exception in LoadCallback.onError", t)
                    }
                }
                return
            }

            // Transition from UNINITIALIZED or ERROR to LOADING
            currentState = State.LOADING
            lastErrorMessage = null
            callback?.let { pendingLoadCallbacks.add(it) }
            shouldSubmit = true
        }

        if (shouldSubmit) {
            try {
                executor.execute {
                    var module: LlmModule? = null
                    try {
                        val modelFile = getModelFile()
                        val tokenizerFile = getTokenizerFile()

                        if (modelFile == null || !modelFile.exists() || modelFile.length() == 0L) {
                            throw IllegalStateException("Model file missing or empty: ${modelFile?.absolutePath}")
                        }
                        if (tokenizerFile == null || !tokenizerFile.exists() || tokenizerFile.length() == 0L) {
                            throw IllegalStateException("Tokenizer file missing or empty: ${tokenizerFile?.absolutePath}")
                        }

                        Log.i(TAG, "Configuring LlmModule (MMAP mode) with model=${modelFile.name}, tokenizer=${tokenizerFile.name}")

                        val config = LlmModuleConfig.create()
                            .modulePath(modelFile.absolutePath)
                            .tokenizerPath(tokenizerFile.absolutePath)
                            .temperature(DEFAULT_TEMPERATURE)
                            .modelType(LlmModuleConfig.MODEL_TYPE_TEXT)
                            .loadMode(LlmModuleConfig.LOAD_MODE_MMAP)
                            .build()

                        // Construct native module
                        module = LlmModule(config)
                        // Load model into memory
                        module.load()

                        val (isClosingOrClosed, callbacksToNotify) = lifecycleLock.withLock {
                            if (currentState == State.CLOSED || isClosing) {
                                val list = ArrayList(pendingLoadCallbacks)
                                pendingLoadCallbacks.clear()
                                Pair(true, list)
                            } else {
                                llmModule = module
                                currentState = State.READY
                                lastErrorMessage = null
                                val list = ArrayList(pendingLoadCallbacks)
                                pendingLoadCallbacks.clear()
                                Pair(false, list)
                            }
                        }

                        if (isClosingOrClosed) {
                            Log.i(TAG, "Model loaded after close was requested; disposing native module immediately.")
                            try {
                                module.close()
                            } catch (closeEx: Exception) {
                                Log.w(TAG, "Exception closing module loaded during close: ${closeEx.message}")
                            }
                            mainHandler.post {
                                for (cb in callbacksToNotify) {
                                    try {
                                        cb.onError("LlamaManager was closed during loading.")
                                    } catch (t: Throwable) {
                                        Log.e(TAG, "Uncaught exception in LoadCallback.onError", t)
                                    }
                                }
                            }
                        } else {
                            Log.i(TAG, "ExecuTorch Llama 3.2 1B model loaded successfully. Notifying ${callbacksToNotify.size} listener(s).")
                            mainHandler.post {
                                for (cb in callbacksToNotify) {
                                    try {
                                        cb.onLoaded()
                                    } catch (t: Throwable) {
                                        Log.e(TAG, "Uncaught exception in LoadCallback.onLoaded", t)
                                    }
                                }
                            }
                        }
                    } catch (t: Throwable) {
                        val err = "Failed to load ExecuTorch model: ${t.message ?: t.javaClass.simpleName}"
                        Log.e(TAG, err, t)

                        // Native resource cleanup: Release any partially created native module
                        try {
                            module?.close()
                        } catch (closeEx: Exception) {
                            Log.w(TAG, "Exception while cleaning up partially initialized LlmModule: ${closeEx.message}")
                        }

                        val callbacksToNotify = lifecycleLock.withLock {
                            if (currentState != State.CLOSED && !isClosing) {
                                currentState = State.ERROR
                                lastErrorMessage = err
                            }
                            llmModule = null
                            val list = ArrayList(pendingLoadCallbacks)
                            pendingLoadCallbacks.clear()
                            list
                        }

                        mainHandler.post {
                            for (cb in callbacksToNotify) {
                                try {
                                    cb.onError(err)
                                } catch (cbErr: Throwable) {
                                    Log.e(TAG, "Uncaught exception in LoadCallback.onError", cbErr)
                                }
                            }
                        }
                    }
                }
            } catch (e: RejectedExecutionException) {
                val (callbacksToNotify, err) = lifecycleLock.withLock {
                    val isTerminatingOrClosed = (currentState == State.CLOSED || isClosing)
                    val errorMsg = if (isTerminatingOrClosed) {
                        "Cannot load model: LlamaManager is closed."
                    } else {
                        "Failed to submit model loading: Executor is terminated or shutting down."
                    }
                    if (!isTerminatingOrClosed) {
                        currentState = State.ERROR
                        lastErrorMessage = errorMsg
                    }
                    val list = ArrayList(pendingLoadCallbacks)
                    pendingLoadCallbacks.clear()
                    Pair(list, errorMsg)
                }

                Log.e(TAG, err, e)

                mainHandler.post {
                    for (cb in callbacksToNotify) {
                        try {
                            cb.onError(err)
                        } catch (cbErr: Throwable) {
                            Log.e(TAG, "Uncaught exception in LoadCallback.onError", cbErr)
                        }
                    }
                }
            }
        }
    }

    /**
     * Executes generation on the dedicated background executor thread.
     *
     * Prevents concurrent generation and validates context requirements before invocation.
     * ExecuTorch's LlmModule.generate() is synchronous on the calling thread. Completion
     * is determined when generate() returns, while tokens are delivered via onResult().
     */
    fun generate(
        prompt: String,
        maxNewTokens: Int = DEFAULT_MAX_NEW_TOKENS,
        temperature: Float = DEFAULT_TEMPERATURE,
        callback: GenerationCallback
    ) {
        // Conservative token estimate (2 chars/token) so non-Latin scripts don't overflow the context.
        val estimatedPromptTokens = prompt.length / 2
        if (estimatedPromptTokens + maxNewTokens > MODEL_CONTEXT_TOKENS) {
            val err = "Prompt too long for $MODEL_CONTEXT_TOKENS-token context (est. $estimatedPromptTokens + $maxNewTokens)."
            Log.w(TAG, err)
            mainHandler.post {
                try {
                    callback.onError(err)
                } catch (t: Throwable) {
                    Log.e(TAG, "Uncaught exception in GenerationCallback.onError", t)
                }
            }
            return
        }

        lifecycleLock.withLock {
            if (currentState != State.READY || llmModule == null || isClosing) {
                val err = when {
                    currentState == State.CLOSED || isClosing -> "LlamaManager is closed."
                    currentState == State.LOADING -> "Model is still loading. Please wait."
                    currentState == State.GENERATING -> "Another inference request is already in progress."
                    currentState == State.ERROR -> "Model failed to load: $lastErrorMessage"
                    currentState == State.UNINITIALIZED -> "Model not loaded. Call loadModel() first."
                    else -> "Module instance unavailable."
                }
                mainHandler.post {
                    try {
                        callback.onError(err)
                    } catch (t: Throwable) {
                        Log.e(TAG, "Uncaught exception in GenerationCallback.onError", t)
                    }
                }
                return
            }
            currentState = State.GENERATING
        }

        try {
            executor.execute {
                val responseBuilder = StringBuilder()
                var callbackErrorCode: Int? = null
                var callbackErrorMessage: String? = null

                try {
                    val module = llmModule ?: throw IllegalStateException("LlmModule released before generation")

                    val genConfig = LlmGenerationConfig.create()
                        .maxNewTokens(maxNewTokens)
                        .temperature(temperature)
                        .echo(false)
                        .build()

                    Log.d(TAG, "Starting synchronous inference (maxNewTokens=$maxNewTokens, temp=$temperature)...")

                    module.generate(prompt, genConfig, object : LlmCallback {
                        override fun onResult(result: String) {
                            responseBuilder.append(result)
                        }

                        override fun onStats(stats: String) {
                            Log.d(TAG, "ExecuTorch generation stats: $stats")
                        }

                        override fun onError(errorCode: Int, message: String) {
                            Log.e(TAG, "ExecuTorch native callback error [$errorCode]: $message")
                            callbackErrorCode = errorCode
                            callbackErrorMessage = message
                        }
                    })

                    // Reset KV-cache context to ensure subsequent generation starts cleanly within the 512-token context budget
                    try {
                        module.resetContext()
                    } catch (e: Exception) {
                        Log.w(TAG, "resetContext encountered exception: ${e.message}")
                    }

                    if (callbackErrorCode != null) {
                        val err = "Native error ($callbackErrorCode): $callbackErrorMessage"
                        lifecycleLock.withLock {
                            if (currentState != State.CLOSED && !isClosing) currentState = State.READY
                        }
                        mainHandler.post {
                            try {
                                callback.onError(err)
                            } catch (t: Throwable) {
                                Log.e(TAG, "Uncaught exception in GenerationCallback.onError", t)
                            }
                        }
                    } else {
                        val finalResponse = responseBuilder.toString().trim()
                        lifecycleLock.withLock {
                            if (currentState != State.CLOSED && !isClosing) currentState = State.READY
                        }
                        mainHandler.post {
                            try {
                                if (finalResponse.isEmpty()) {
                                    // Blank output is a failure, so the caller falls back to the template.
                                    Log.w(TAG, "Model returned an empty response.")
                                    callback.onError("Model returned an empty response.")
                                } else {
                                    callback.onSuccess(finalResponse)
                                }
                            } catch (t: Throwable) {
                                Log.e(TAG, "Uncaught exception in GenerationCallback result dispatch", t)
                            }
                        }
                    }
                } catch (t: Throwable) {
                    val err = "Generation failed: ${t.message ?: t.javaClass.simpleName}"
                    Log.e(TAG, err, t)
                    lifecycleLock.withLock {
                        if (currentState != State.CLOSED && !isClosing) currentState = State.READY
                    }
                    mainHandler.post {
                        try {
                            callback.onError(err)
                        } catch (cbErr: Throwable) {
                            Log.e(TAG, "Uncaught exception in GenerationCallback.onError", cbErr)
                        }
                    }
                }
            }
        } catch (e: RejectedExecutionException) {
            val err = lifecycleLock.withLock {
                val isTerminatingOrClosed = (currentState == State.CLOSED || isClosing)
                if (!isTerminatingOrClosed) {
                    currentState = State.READY
                    "Failed to submit inference: Executor is terminated or shutting down."
                } else {
                    "Cannot execute inference: LlamaManager is closed."
                }
            }
            Log.e(TAG, err, e)
            mainHandler.post {
                try {
                    callback.onError(err)
                } catch (t: Throwable) {
                    Log.e(TAG, "Uncaught exception in GenerationCallback.onError", t)
                }
            }
        }
    }

    /**
     * Safely closes native resources and terminates the background executor.
     * Truly idempotent: repeated calls queue close callbacks without submitting new tasks
     * to a terminated executor or throwing RejectedExecutionException.
     */
    fun close(onClosed: (() -> Unit)? = null) {
        // Discard singleton reference immediately if closing this instance
        synchronized(Companion) {
            if (instance === this@LlamaManager) {
                instance = null
            }
        }

        var shouldSubmitCloseTask = false
        lifecycleLock.withLock {
            if (currentState == State.CLOSED) {
                mainHandler.post {
                    try {
                        onClosed?.invoke()
                    } catch (t: Throwable) {
                        Log.e(TAG, "Uncaught exception in close callback", t)
                    }
                }
                return
            }

            onClosed?.let { pendingCloseCallbacks.add(it) }

            if (!isClosing) {
                isClosing = true
                shouldSubmitCloseTask = true
            }
        }

        if (shouldSubmitCloseTask) {
            try {
                executor.execute {
                    val (loadCallbacks, closeCallbacks) = lifecycleLock.withLock {
                        currentState = State.CLOSED
                        val loads = ArrayList(pendingLoadCallbacks)
                        pendingLoadCallbacks.clear()

                        val closes = ArrayList(pendingCloseCallbacks)
                        pendingCloseCallbacks.clear()

                        try {
                            llmModule?.close()
                        } catch (e: Exception) {
                            Log.w(TAG, "Exception closing LlmModule: ${e.message}")
                        } finally {
                            llmModule = null
                        }
                        Pair(loads, closes)
                    }

                    mainHandler.post {
                        for (cb in loadCallbacks) {
                            try {
                                cb.onError("LlamaManager was closed during loading.")
                            } catch (t: Throwable) {
                                Log.e(TAG, "Uncaught exception in LoadCallback.onError", t)
                            }
                        }
                    }

                    executor.shutdown()
                    Log.i(TAG, "LlamaManager closed and executor terminated.")

                    mainHandler.post {
                        for (cb in closeCallbacks) {
                            try {
                                cb.invoke()
                            } catch (t: Throwable) {
                                Log.e(TAG, "Uncaught exception in close callback", t)
                            }
                        }
                    }
                }
            } catch (e: RejectedExecutionException) {
                // If executor is already shutting down or rejects execution, perform synchronous cleanup
                val (loadCallbacks, closeCallbacks) = lifecycleLock.withLock {
                    currentState = State.CLOSED
                    val loads = ArrayList(pendingLoadCallbacks)
                    pendingLoadCallbacks.clear()

                    val closes = ArrayList(pendingCloseCallbacks)
                    pendingCloseCallbacks.clear()

                    try {
                        llmModule?.close()
                    } catch (ex: Exception) {
                        Log.w(TAG, "Exception closing LlmModule: ${ex.message}")
                    } finally {
                        llmModule = null
                    }
                    Pair(loads, closes)
                }

                mainHandler.post {
                    for (cb in loadCallbacks) {
                        try {
                            cb.onError("LlamaManager was closed during loading.")
                        } catch (t: Throwable) {
                            Log.e(TAG, "Uncaught exception in LoadCallback.onError", t)
                        }
                    }
                    for (cb in closeCallbacks) {
                        try {
                            cb.invoke()
                        } catch (t: Throwable) {
                            Log.e(TAG, "Uncaught exception in close callback", t)
                        }
                    }
                }
            }
        }
    }
}
