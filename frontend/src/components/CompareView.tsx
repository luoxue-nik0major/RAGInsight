import FlowChart from './FlowChart';
import type { Session, RootCause, PerturbationResult } from '../types';

interface CompareSession {
  label: string;
  session: Session | null;
  rootCause: RootCause | null;
  perturbationResults: PerturbationResult[] | null;
}

interface Props {
  sessions: CompareSession[];
  onNodeClick: (step: any) => void;
}

export default function CompareView({ sessions, onNodeClick }: Props) {
  return (
    <div className="h-full w-full flex bg-gray-50">
      {sessions.map((item, idx) => (
        <div
          key={idx}
          className={`flex-1 min-w-0 ${idx > 0 ? 'border-l border-gray-300' : ''} flex flex-col`}
        >
          <div className="px-3 py-2 bg-white border-b border-gray-200 flex items-center justify-between">
            <span className="text-sm font-semibold text-gray-700">{item.label}</span>
            {item.session && (
              <span className="text-xs text-gray-400">Session #{item.session.id}</span>
            )}
          </div>
          <div className="flex-1 min-h-0">
            <FlowChart
              session={item.session}
              rootCause={item.rootCause}
              perturbationResults={item.perturbationResults}
              onNodeClick={onNodeClick}
            />
          </div>
        </div>
      ))}
    </div>
  );
}
