/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import {
  GenerateContentResponse
} from '@google/genai';
import type {
  GenerateContentParameters,
  CountTokensParameters,
  EmbedContentParameters,
  Content,

  CountTokensResponse,
  EmbedContentResponse,
  FinishReason} from '@google/genai';
import type { ContentGenerator } from './contentGenerator.js';

interface OpenAIMessage {
  role: string;
  content: string;
}

interface OpenAICompletionResponse {
  id: string;
  object: string;
  created: number;
  choices: Array<{
    index: number;
    message: {
      role: string;
      content: string;
    };
    finish_reason: string;
  }>;
  usage: {
    prompt_tokens: number;
    completion_tokens: number;
    total_tokens: number;
  };
}

interface OpenAIStreamChunk {
  id: string;
  object: string;
  created: number;
  model: string;
  choices: Array<{
    index: number;
    delta: {
      content?: string;
    };
    finish_reason: string | null;
  }>;
}

export class OllamaContentGenerator implements ContentGenerator {
  private baseUrl: string;
  private model: string;

  constructor(
    baseUrl: string = 'http://localhost:11434/v1',
    model: string = 'llama3',
  ) {
    this.baseUrl = baseUrl;
    this.model = model;
  }

  private toOpenAIRole(role: string): string {
    if (role === 'user') return 'user';
    if (role === 'model') return 'assistant';
    return 'system';
  }

  private normalizeContents(contents: unknown): Content[] {
    if (Array.isArray(contents)) {
      return contents as Content[];
    }
    if (typeof contents === 'string') {
      return [{ role: 'user', parts: [{ text: contents }] }];
    }
    // Assume it's a single Content object
    return [contents as Content];
  }

  private toOpenAIMessages(contents: Content[]): OpenAIMessage[] {
    return contents.map((content) => {
      const parts = content.parts || [];
      const text = parts
        .map((part) => {
          if (part.text) return part.text;
          return '';
        })
        .join('');
      return {
        role: this.toOpenAIRole(content.role || 'user'),
        content: text,
      };
    });
  }

  async generateContent(
    request: GenerateContentParameters,
    _userPromptId: string,
  ): Promise<GenerateContentResponse> {
    const contents = this.normalizeContents(request.contents);
    const messages = this.toOpenAIMessages(contents);
    const model = request.model.startsWith('gemini')
      ? this.model
      : request.model;

    const response = await fetch(`${this.baseUrl}/chat/completions`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        model,
        messages,
        stream: false,
      }),
    });

    if (!response.ok) {
      const errorText = await response.text();
      throw new Error(`Ollama API error: ${response.status} - ${errorText}`);
    }

    const data = (await response.json()) as OpenAICompletionResponse;

    const result = {
      candidates: data.choices.map((choice) => ({
        content: {
          role: 'model',
          parts: [{ text: choice.message.content }],
        },
        finishReason: choice.finish_reason.toUpperCase() as FinishReason,
        index: choice.index,
      })),
      usageMetadata: {
        promptTokenCount: data.usage.prompt_tokens,
        candidatesTokenCount: data.usage.completion_tokens,
        totalTokenCount: data.usage.total_tokens,
      },
    } as unknown as GenerateContentResponse;

    Object.setPrototypeOf(result, GenerateContentResponse.prototype);
    return result;
  }

  async generateContentStream(
    request: GenerateContentParameters,
    _userPromptId: string,
  ): Promise<AsyncGenerator<GenerateContentResponse>> {
    const contents = this.normalizeContents(request.contents);
    const messages = this.toOpenAIMessages(contents);
    const model = request.model.startsWith('gemini')
      ? this.model
      : request.model;

    const response = await fetch(`${this.baseUrl}/chat/completions`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        model,
        messages,
        stream: true,
      }),
    });

    if (!response.ok) {
      const errorText = await response.text();
      throw new Error(`Ollama API error: ${response.status} - ${errorText}`);
    }

    if (!response.body) {
      throw new Error('No response body from Ollama');
    }

    const reader = response.body.getReader();
    const decoder = new TextDecoder();

    return (async function* () {
      let buffer = '';
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\n');
        buffer = lines.pop() || '';

        for (const line of lines) {
          const trimmed = line.trim();
          if (!trimmed || trimmed === 'data: [DONE]') continue;
          if (trimmed.startsWith('data: ')) {
            try {
              const data = JSON.parse(trimmed.slice(6)) as OpenAIStreamChunk;
              if (data.choices && data.choices.length > 0) {
                const delta = data.choices[0].delta;
                if (delta.content) {
                  const chunk = {
                    candidates: [
                      {
                        content: {
                          role: 'model',
                          parts: [{ text: delta.content }],
                        },
                        finishReason:
                          (data.choices[0].finish_reason?.toUpperCase() as FinishReason) ||
                          undefined,
                        index: 0,
                      },
                    ],
                  } as unknown as GenerateContentResponse;
                  Object.setPrototypeOf(
                    chunk,
                    GenerateContentResponse.prototype,
                  );
                  yield chunk;
                }
              }
            } catch (_e) {
              // Ignore parsing errors for individual chunks
            }
          }
        }
      }
    })();
  }

  async countTokens(
    _request: CountTokensParameters,
  ): Promise<CountTokensResponse> {
    return {
      totalTokens: 0,
    };
  }

  async embedContent(
    request: EmbedContentParameters,
  ): Promise<EmbedContentResponse> {
    const model = request.model.startsWith('gemini')
      ? this.model
      : request.model;
    const contents = this.normalizeContents(request.contents);

    // Flatten texts from all parts of all contents
    const texts: string[] = [];
    for (const content of contents) {
      if (content.parts) {
        for (const part of content.parts) {
          if (part.text) {
            texts.push(part.text);
          }
        }
      }
    }

    const embeddings: Array<{ values: number[] }> = [];

    for (const input of texts) {
      const response = await fetch(`${this.baseUrl}/embeddings`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          model,
          input,
        }),
      });

      if (!response.ok) {
        throw new Error(`Ollama Embedding Error: ${response.statusText}`);
      }

      const data = (await response.json());
      if (data.data && data.data.length > 0) {
        embeddings.push({ values: data.data[0].embedding });
      }
    }

    return {
      embeddings,
    };
  }
}
