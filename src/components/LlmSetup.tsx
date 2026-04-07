import React, { useState, useCallback } from 'react';
import { Box, Text } from '../ink.js';
import { Dialog } from './design-system/Dialog.js';
import { Select } from './CustomSelect/select.js';
import { TextInput } from './TextInput.js';
import type { LlmProvider, LlmConfig } from '../utils/config.js';
import { saveGlobalConfig, getGlobalConfig } from '../utils/config.js';

interface LlmSetupProps {
  onComplete: (config: LlmConfig) => void;
  onSkip?: () => void;
}

const PROVIDER_OPTIONS: { value: LlmProvider; label: string; baseUrl: string }[] = [
  { value: 'anthropic', label: 'Anthropic', baseUrl: 'https://api.anthropic.com' },
  { value: 'openai', label: 'OpenAI', baseUrl: 'https://api.openai.com/v1' },
  { value: 'openrouter', label: 'OpenRouter', baseUrl: 'https://openrouter.ai/api/v1' },
  { value: 'google', label: 'Google Vertex AI', baseUrl: 'https://us-central1-aiplatform.googleapis.com' },
  { value: 'azure', label: 'Azure Foundry', baseUrl: '' },
  { value: 'aws', label: 'AWS Bedrock', baseUrl: '' },
  { value: 'minimax', label: 'MiniMax', baseUrl: '' },
  { value: 'ollama', label: 'Ollama (Local)', baseUrl: 'http://localhost:11434' },
  { value: 'custom', label: 'Custom Endpoint', baseUrl: '' },
];

export function LlmSetup({ onComplete, onSkip }: LlmSetupProps): React.ReactNode {
  const [provider, setProvider] = useState<LlmProvider>('anthropic');
  const [baseUrl, setBaseUrl] = useState('');
  const [apiKey, setApiKey] = useState('');
  const [model, setModel] = useState('');
  const [error, setError] = useState('');

  const selectedProvider = PROVIDER_OPTIONS.find(p => p.value === provider);

  const handleSubmit = useCallback(() => {
    if (!apiKey.trim() && provider !== 'ollama') {
      setError('API Key is required');
      return;
    }
    if (!model.trim() && provider !== 'ollama') {
      setError('Model is required');
      return;
    }

    const config: LlmConfig = {
      provider,
      baseUrl: baseUrl || selectedProvider?.baseUrl,
      model,
    };

    const globalConfig = getGlobalConfig();
    saveGlobalConfig({
      ...globalConfig,
      llmConfig: config,
    });

    if (apiKey) {
      if (provider === 'anthropic') {
        process.env.ANTHROPIC_API_KEY = apiKey;
      } else if (provider === 'openai') {
        process.env.OPENAI_API_KEY = apiKey;
      } else if (provider === 'openrouter') {
        process.env.OPENROUTER_API_KEY = apiKey;
      }
    }

    onComplete(config);
  }, [provider, baseUrl, apiKey, model, selectedProvider, onComplete]);

  return (
    <Dialog title="Configure LLM Provider" onCancel={onSkip}>
      <Box flexDirection="column" gap={1}>
        <Box flexDirection="column" gap={1}>
          <Text bold>Provider</Text>
          <Select
            options={PROVIDER_OPTIONS.map(p => ({ value: p.value, label: p.label }))}
            value={provider}
            onChange={(value) => {
              setProvider(value as LlmProvider);
              const opt = PROVIDER_OPTIONS.find(p => p.value === value);
              setBaseUrl(opt?.baseUrl || '');
              setError('');
            }}
          />
        </Box>

        <Box flexDirection="column" gap={1}>
          <Text bold>Base URL (optional)</Text>
          <TextInput
            value={baseUrl}
            onChange={setBaseUrl}
            placeholder={selectedProvider?.baseUrl || 'https://api.example.com/v1'}
          />
          <Text dimColor>Leave empty to use provider default</Text>
        </Box>

        {provider !== 'ollama' && (
          <Box flexDirection="column" gap={1}>
            <Text bold>API Key</Text>
            <TextInput
              value={apiKey}
              onChange={(value) => {
                setApiKey(value);
                setError('');
              }}
              placeholder="Enter your API key"
              secureTextEntry
            />
          </Box>
        )}

        {provider !== 'ollama' && (
          <Box flexDirection="column" gap={1}>
            <Text bold>Model</Text>
            <TextInput
              value={model}
              onChange={(value) => {
                setModel(value);
                setError('');
              }}
              placeholder={getDefaultModel(provider)}
            />
          </Box>
        )}

        {error && (
          <Text color="red">{error}</Text>
        )}

        <Box marginTop={1} justifyContent="flex-end" gap={2}>
          {onSkip && (
            <Text bold dimColor>
              [Skip]
            </Text>
          )}
        </Box>
      </Box>
    </Dialog>
  );
}

function getDefaultModel(provider: LlmProvider): string {
  switch (provider) {
    case 'anthropic':
      return 'claude-sonnet-4-20250514';
    case 'openai':
      return 'gpt-4o';
    case 'openrouter':
      return 'anthropic/claude-3.5-sonnet';
    case 'google':
      return 'claude-3-5-sonnet-v2@20240620';
    case 'azure':
      return 'claude-3-5-sonnet';
    case 'aws':
      return 'anthropic.claude-3-5-sonnet-20241022-v2:0';
    case 'minimax':
      return 'MiniMax-M2.7-highspeed';
    default:
      return '';
  }
}
