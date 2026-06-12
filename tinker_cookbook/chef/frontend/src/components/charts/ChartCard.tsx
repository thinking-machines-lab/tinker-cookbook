/** Shared chart card wrapper — consistent styling + optional collapse toggle. */

import { useState } from 'react';

interface Props {
  title: string;
  subtitle?: string;
  collapsible?: boolean;
  defaultCollapsed?: boolean;
  /** Extra controls to render in the header (e.g., smoothing buttons). */
  headerRight?: React.ReactNode;
  children: React.ReactNode;
}

export function ChartCard({
  title,
  subtitle,
  collapsible = false,
  defaultCollapsed = false,
  headerRight,
  children,
}: Props) {
  const [collapsed, setCollapsed] = useState(defaultCollapsed);

  return (
    <div className="card" style={{ marginBottom: '0.75rem' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', gap: '0.5rem', marginBottom: collapsed ? 0 : '0.5rem' }}>
        <div
          className="card-title"
          onClick={collapsible ? () => setCollapsed(!collapsed) : undefined}
          style={{ cursor: collapsible ? 'pointer' : 'default', userSelect: 'none', marginBottom: 0 }}
        >
          {collapsible && (
            <span style={{ fontSize: '0.625rem', marginRight: '0.25rem' }}>{collapsed ? '\u25b6' : '\u25bc'}</span>
          )}
          {title}
          {subtitle && (
            <span style={{ fontSize: '0.5625rem', color: 'var(--text-muted)', marginLeft: '0.375rem', fontWeight: 400 }}>
              {subtitle}
            </span>
          )}
        </div>
        {!collapsed && headerRight}
      </div>
      {!collapsed && children}
    </div>
  );
}
