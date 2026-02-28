import { createContext, useContext, useState, useCallback, type ReactNode } from "react";
import type { ChouchanePhase } from "@/services/chouchaneApi";

interface ChouchaneSessionContextValue {
  sessionId: string | null;
  setSessionId: (id: string | null) => void;
  phase: ChouchanePhase;
  setPhase: (phase: ChouchanePhase) => void;
}

const ChouchaneSessionContext = createContext<ChouchaneSessionContextValue | null>(null);

export function ChouchaneSessionProvider({ children }: { children: ReactNode }) {
  const [sessionId, setSessionIdState] = useState<string | null>(null);
  const [phase, setPhaseState] = useState<ChouchanePhase>("yasmine");

  const setSessionId = useCallback((id: string | null) => {
    setSessionIdState(id);
  }, []);

  const setPhase = useCallback((p: ChouchanePhase) => {
    setPhaseState(p);
  }, []);

  return (
    <ChouchaneSessionContext.Provider
      value={{ sessionId, setSessionId, phase, setPhase }}
    >
      {children}
    </ChouchaneSessionContext.Provider>
  );
}

export function useChouchaneSession() {
  const ctx = useContext(ChouchaneSessionContext);
  if (!ctx) {
    throw new Error("useChouchaneSession must be used within ChouchaneSessionProvider");
  }
  return ctx;
}
