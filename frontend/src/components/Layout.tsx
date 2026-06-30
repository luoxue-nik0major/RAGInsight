import { ReactNode } from 'react';

interface Props {
  leftPanel: ReactNode;
  centerPanel: ReactNode;
  rightPanel: ReactNode;
}

export default function Layout({ leftPanel, centerPanel, rightPanel }: Props) {
  return (
    <div className="flex h-screen w-screen overflow-hidden bg-white">
      <div className="w-80 flex-shrink-0 h-full">{leftPanel}</div>
      <div className="flex-1 h-full min-w-0">{centerPanel}</div>
      <div className="w-96 flex-shrink-0 h-full">{rightPanel}</div>
    </div>
  );
}
