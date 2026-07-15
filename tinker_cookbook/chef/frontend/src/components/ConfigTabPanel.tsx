import { useState } from 'react';

export function ConfigTabPanel({ config }: { config: Record<string, unknown> }) {
  const [search, setSearch] = useState('');
  const configStr = JSON.stringify(config, null, 2);
  const lines = configStr.split('\n');
  const filtered = search
    ? lines.filter((line) => line.toLowerCase().includes(search.toLowerCase()))
    : lines;

  const topLevelKeys = Object.keys(config);
  const scalarKeys = topLevelKeys.filter((k) => typeof config[k] !== 'object' || config[k] === null);
  const objectKeys = topLevelKeys.filter((k) => typeof config[k] === 'object' && config[k] !== null);
  const [expandedSections, setExpandedSections] = useState<Set<string>>(new Set());

  return (
    <div>
      <div className="card" style={{ marginBottom: '0.75rem' }}>
        <div className="card-header">
          <span>Configuration</span>
          <input type="text" placeholder="Search..." value={search} onChange={(e) => setSearch(e.target.value)} style={{ width: '180px' }} />
        </div>
        {search ? (
          <pre className="mono" style={{ whiteSpace: 'pre-wrap', lineHeight: 1.6, fontSize: '0.6875rem' }}>{filtered.join('\n')}</pre>
        ) : (
          <div style={{ fontSize: '0.8125rem' }}>
            <table style={{ marginBottom: '0.75rem' }}>
              <tbody>
                {scalarKeys.map((key) => (
                  <tr key={key} style={{ cursor: 'default' }}>
                    <td className="mono text-muted" style={{ width: '200px', fontSize: '0.75rem' }}>{key}</td>
                    <td className="mono" style={{ fontSize: '0.75rem' }}>{String(config[key] ?? 'null')}</td>
                  </tr>
                ))}
              </tbody>
            </table>
            {objectKeys.map((key) => (
              <div key={key} style={{ marginBottom: '0.375rem' }}>
                <div
                  onClick={() => setExpandedSections((prev) => { const n = new Set(prev); if (n.has(key)) n.delete(key); else n.add(key); return n; })}
                  style={{ cursor: 'pointer', padding: '0.375rem 0.5rem', background: 'var(--bg-elevated)', borderRadius: '4px', fontFamily: 'var(--font-mono)', fontSize: '0.75rem', fontWeight: 600, color: 'var(--text-secondary)', display: 'flex', alignItems: 'center', gap: '0.375rem' }}
                >
                  <span>{expandedSections.has(key) ? '\u25bc' : '\u25b6'}</span>
                  <span>{key}</span>
                  <span className="text-muted" style={{ fontWeight: 400, fontSize: '0.625rem' }}>
                    {Array.isArray(config[key]) ? `[${(config[key] as unknown[]).length} items]` : `{${Object.keys(config[key] as object).length} fields}`}
                  </span>
                </div>
                {expandedSections.has(key) && (
                  <pre className="mono" style={{ whiteSpace: 'pre-wrap', lineHeight: 1.5, fontSize: '0.6875rem', padding: '0.5rem', margin: '0.25rem 0 0 1rem', background: 'var(--bg-elevated)', borderRadius: '4px', maxHeight: '300px', overflow: 'auto' }}>
                    {JSON.stringify(config[key], null, 2)}
                  </pre>
                )}
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
