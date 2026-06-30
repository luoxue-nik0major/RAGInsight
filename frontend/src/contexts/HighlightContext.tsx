import { createContext, useContext, useState, useCallback, type ReactNode } from 'react';

interface HighlightState {
  /** chunk indices currently highlighted from answer hover */
  highlightedChunkIndices: Set<number>;
  /** answer segment index currently highlighted from chunk hover */
  highlightedSegmentIndex: number | null;
}

interface HighlightActions {
  /** Called when hovering over answer segment — highlights linked chunks */
  highlightChunks: (chunkIndices: number[]) => void;
  /** Called when hovering over chunk node — highlights linked answer segment */
  highlightAnswerSegment: (segmentIndex: number | null) => void;
  /** Clear all highlights */
  clearHighlights: () => void;
}

type HighlightContextType = HighlightState & HighlightActions;

const HighlightContext = createContext<HighlightContextType>({
  highlightedChunkIndices: new Set(),
  highlightedSegmentIndex: null,
  highlightChunks: () => {},
  highlightAnswerSegment: () => {},
  clearHighlights: () => {},
});

export function HighlightProvider({ children }: { children: ReactNode }) {
  const [highlightedChunkIndices, setHighlightedChunkIndices] = useState<Set<number>>(new Set());
  const [highlightedSegmentIndex, setHighlightedSegmentIndex] = useState<number | null>(null);

  const highlightChunks = useCallback((chunkIndices: number[]) => {
    setHighlightedChunkIndices(new Set(chunkIndices));
    setHighlightedSegmentIndex(null);
  }, []);

  const highlightAnswerSegment = useCallback((segmentIndex: number | null) => {
    setHighlightedSegmentIndex(segmentIndex);
    setHighlightedChunkIndices(new Set());
  }, []);

  const clearHighlights = useCallback(() => {
    setHighlightedChunkIndices(new Set());
    setHighlightedSegmentIndex(null);
  }, []);

  return (
    <HighlightContext.Provider value={{
      highlightedChunkIndices,
      highlightedSegmentIndex,
      highlightChunks,
      highlightAnswerSegment,
      clearHighlights,
    }}>
      {children}
    </HighlightContext.Provider>
  );
}

export function useHighlightContext() {
  return useContext(HighlightContext);
}
