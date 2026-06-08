import { ReactNode } from "react";

interface CardProps {
  children: ReactNode;
  className?: string;
}

export function Card({ children, className = "" }: CardProps) {
  return (
    <div className={`bg-slate-800 border border-slate-700 rounded-lg p-5 ${className}`}>
      {children}
    </div>
  );
}

export function CardTitle({ children, className = "" }: CardProps) {
  return (
    <h3 className={`text-xs font-semibold uppercase tracking-widest text-slate-400 mb-3 ${className}`}>
      {children}
    </h3>
  );
}
