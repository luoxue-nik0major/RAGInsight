import { useCallback, useRef } from 'react';
import type { SSEEvent } from '../types';

export function useSSE() {
  const abortRef = useRef<AbortController | null>(null);

  const connect = useCallback((
    query: string,
    strategy: string,
    onEvent: (event: SSEEvent) => void,
    onDone: () => void,
    onError: (err: string) => void,
    collection?: string,
  ) => {
    // Abort previous
    if (abortRef.current) {
      abortRef.current.abort();
    }
    const controller = new AbortController();
    abortRef.current = controller;

    const body: Record<string, string> = { query, strategy };
    if (collection) {
      body.collection = collection;
    }
    fetch('/api/query', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
      signal: controller.signal,
    }).then(async (response) => {
      const reader = response.body?.getReader();
      const decoder = new TextDecoder();
      let buffer = '';

      if (!reader) {
        onError('No response body');
        return;
      }

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\n\n');
        buffer = lines.pop() || '';

        for (const line of lines) {
          const trimmed = line.trim();
          if (!trimmed.startsWith('data: ')) continue;
          const dataStr = trimmed.slice(6);
          if (dataStr === '[DONE]') {
            onDone();
            return;
          }
          try {
            const parsed = JSON.parse(dataStr);
            onEvent(parsed);
          } catch (e) {
            console.error('Parse SSE data failed:', dataStr);
          }
        }
      }
      onDone();
    }).catch((err) => {
      if (err.name !== 'AbortError') {
        onError(err.message || 'Unknown error');
      }
    });

    return () => {
      controller.abort();
    };
  }, []);

  const disconnect = useCallback(() => {
    if (abortRef.current) {
      abortRef.current.abort();
      abortRef.current = null;
    }
  }, []);

  return { connect, disconnect };
}
