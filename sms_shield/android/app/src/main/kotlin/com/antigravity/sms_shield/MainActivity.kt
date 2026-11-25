package com.antigravity.sms_shield

import android.util.Log
import io.flutter.embedding.android.FlutterActivity
import io.flutter.embedding.engine.FlutterEngine
import io.flutter.plugin.common.MethodChannel
import kotlin.math.min

class MainActivity : FlutterActivity() {

    companion object {
        private const val TAG = "MainActivity"
        private const val CHANNEL = "com.antigravity.sms_shield/llama_executorch"
        // Keep in sync with LlamaManager (512-token context: prompt + new tokens must fit).
        private const val MAX_GENERATION_TOKENS = 96
    }

    override fun configureFlutterEngine(flutterEngine: FlutterEngine) {
        super.configureFlutterEngine(flutterEngine)

        MethodChannel(flutterEngine.dartExecutor.binaryMessenger, CHANNEL).setMethodCallHandler { call, result ->
            // Fetch per call so a closed/reset singleton is never reused after the Activity is recreated.
            val manager = LlamaManager.getInstance(applicationContext)
            when (call.method) {
                "isModelReady" -> {
                    if (manager.isModelReady()) {
                        result.success(true)
                    } else if (manager.areModelFilesPresent()) {
                        // Start/await background load and notify Flutter when ready
                        manager.loadModel(object : LlamaManager.LoadCallback {
                            override fun onLoaded() {
                                result.success(true)
                            }

                            override fun onError(error: String) {
                                Log.w(TAG, "isModelReady load check failed: $error")
                                result.success(false)
                            }
                        })
                    } else {
                        // Model or tokenizer file missing in app external files directory
                        result.success(false)
                    }
                }

                "generate" -> {
                    val systemPrompt = call.argument<String>("systemPrompt") ?: ""
                    val rawUserPrompt = call.argument<String>("userPrompt")
                        ?: call.argument<String>("prompt")
                        ?: ""
                    val requestedTokens = call.argument<Int>("maxNewTokens") ?: MAX_GENERATION_TOKENS
                    val temperature = (call.argument<Double>("temperature") ?: 0.1).toFloat()

                    // Cap new tokens so prompt + output stay inside the 512-token context
                    val safeTokens = min(requestedTokens, MAX_GENERATION_TOKENS).coerceAtLeast(1)

                    if (!manager.areModelFilesPresent()) {
                        result.error(
                            "MODEL_NOT_FOUND",
                            "Model artifacts missing. Place llama1b_q4.pte and tokenizer.model in external files dir.",
                            null
                        )
                        return@setMethodCallHandler
                    }

                    val formattedPrompt = formatLlamaPrompt(systemPrompt, rawUserPrompt)

                    val executeGeneration = {
                        manager.generate(
                            prompt = formattedPrompt,
                            maxNewTokens = safeTokens,
                            temperature = temperature,
                            callback = object : LlamaManager.GenerationCallback {
                                override fun onSuccess(response: String) {
                                    result.success(response)
                                }

                                override fun onError(error: String) {
                                    result.error("GENERATION_ERROR", error, null)
                                }
                            }
                        )
                    }

                    if (manager.isModelReady()) {
                        executeGeneration()
                    } else {
                        // Ensure model is loaded first before generating
                        manager.loadModel(object : LlamaManager.LoadCallback {
                            override fun onLoaded() {
                                executeGeneration()
                            }

                            override fun onError(error: String) {
                                result.error("MODEL_LOAD_ERROR", error, null)
                            }
                        })
                    }
                }

                else -> {
                    result.notImplemented()
                }
            }
        }
    }

    /**
     * Formats prompt into the standard Meta Llama 3.2 Instruct chat template:
     * <|begin_of_text|><|start_header_id|>system<|end_header_id|>
     * {systemPrompt}<|eot_id|><|start_header_id|>user<|end_header_id|>
     * {userPrompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
     */
    private fun formatLlamaPrompt(systemPrompt: String, userPrompt: String): String {
        if (userPrompt.startsWith("<|begin_of_text|>")) {
            return userPrompt
        }
        val sb = StringBuilder("<|begin_of_text|>")
        if (systemPrompt.isNotBlank()) {
            sb.append("<|start_header_id|>system<|end_header_id|>\n\n")
            sb.append(systemPrompt.trim())
            sb.append("<|eot_id|>")
        }
        sb.append("<|start_header_id|>user<|end_header_id|>\n\n")
        sb.append(userPrompt.trim())
        sb.append("<|eot_id|>")
        sb.append("<|start_header_id|>assistant<|end_header_id|>\n\n")
        return sb.toString()
    }

    override fun onDestroy() {
        super.onDestroy()
        // If the activity is finishing permanently (not just configuration change/rotation),
        // gracefully release native module resources and worker threads.
        if (isFinishing) {
            LlamaManager.resetInstance()
        }
    }
}
