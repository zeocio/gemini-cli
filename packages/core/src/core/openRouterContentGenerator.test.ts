/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import { describe, it, expect, vi, beforeEach, afterAll } from 'vitest';
import { createOpenRouterContentGenerator } from './openRouterContentGenerator.js';
import { ContentGeneratorConfig, AuthType } from './contentGenerator.js';
import { Config } from '../config/config.js';

vi.mock('../code_assist/codeAssist.js');
vi.mock('@google/genai');

describe('OpenRouter Content Generator', () => {
  const mockConfig: ContentGeneratorConfig = {
    model: 'google/gemini-2.5-flash',
    apiKey: 'test-api-key',
    authType: AuthType.USE_OPENROUTER,
    openRouterBaseUrl: 'https://openrouter.ai/api/v1',
  };

  const mockHttpOptions = {
    headers: {
      'User-Agent': 'test-user-agent',
    },
  };

  it('should create content generator with correct configuration', () => {
    const generator = createOpenRouterContentGenerator(
      mockConfig,
      mockHttpOptions,
    );

    expect(generator).toBeDefined();
    expect(generator.generateContent).toBeDefined();
    expect(generator.generateContentStream).toBeDefined();
    expect(generator.countTokens).toBeDefined();
    expect(generator.embedContent).toBeDefined();
  });

  it('should estimate tokens correctly', async () => {
    const generator = createOpenRouterContentGenerator(
      mockConfig,
      mockHttpOptions,
    );

    const result = await generator.countTokens({
      model: 'google/gemini-2.5-flash',
      contents: [
        {
          role: 'user',
          parts: [{ text: 'Hello, how are you?' }],
        },
      ],
    });

    // "Hello, how are you?" is 19 characters, estimated as ~5 tokens
    expect(result.totalTokens).toBe(5);
    expect(result.cachedContentTokenCount).toBe(0);
  });

  it('should handle string content in countTokens', async () => {
    const generator = createOpenRouterContentGenerator(
      mockConfig,
      mockHttpOptions,
    );

    const result = await generator.countTokens({
      model: 'google/gemini-2.5-flash',
      contents: 'Hello world',
    });

    // "Hello world" is 11 characters, estimated as ~3 tokens
    expect(result.totalTokens).toBe(3);
  });

  it('should throw error for embedContent', async () => {
    const generator = createOpenRouterContentGenerator(
      mockConfig,
      mockHttpOptions,
    );

    await expect(
      generator.embedContent({
        model: 'text-embedding-ada-002',
        contents: ['test'],
      }),
    ).rejects.toThrow(
      'Embeddings are not supported through OpenRouter for Gemini models',
    );
  });
});

describe('Model mapping', () => {
  const originalEnv = process.env;
  const mockConfig = {
      getModel: vi.fn().mockReturnValue('gemini-2.5-flash'),
      setModel: vi.fn(),
      flashFallbackHandler: vi.fn(),
      getProxy: vi.fn(),
    } as unknown as Config;

  beforeEach(() => {
      // Reset modules to re-evaluate imports and environment variables
      vi.resetModules();
      // Restore process.env before each test
      process.env = { ...originalEnv };
      vi.clearAllMocks();
  });

  afterAll(() => {
      // Restore original process.env after all tests
      process.env = originalEnv;
  });

  it('should map Gemini models to OpenRouter format', async () => {
    // Set the environment variable for the test
    process.env.OPENROUTER_API_KEY = 'test-key';

    const { createContentGeneratorConfig } = await import(
      './contentGenerator.js'
    );

    const config = await createContentGeneratorConfig(
      mockConfig,
      AuthType.USE_OPENROUTER,
    );

    expect(config.model).toBe('google/gemini-2.5-flash');

    // Clean up
    delete process.env.OPENROUTER_API_KEY;
  });
});
