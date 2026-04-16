import { useState, useMemo } from 'react';

interface Column<T> {
  key: string;
  label: string;
  render: (row: T) => React.ReactNode;
  sortValue?: (row: T) => number | string;
}

interface Props<T> {
  columns: Column<T>[];
  data: T[];
  onRowClick?: (row: T) => void;
  rowKey: (row: T) => string;
}

export function SortableTable<T>({ columns, data, onRowClick, rowKey }: Props<T>) {
  const [sortKey, setSortKey] = useState<string | null>(null);
  const [sortDir, setSortDir] = useState<'asc' | 'desc'>('asc');

  const sorted = useMemo(() => {
    if (!sortKey) return data;
    const col = columns.find((c) => c.key === sortKey);
    if (!col?.sortValue) return data;
    const getValue = col.sortValue;
    return [...data].sort((a, b) => {
      const va = getValue(a);
      const vb = getValue(b);
      const cmp = va < vb ? -1 : va > vb ? 1 : 0;
      return sortDir === 'asc' ? cmp : -cmp;
    });
  }, [data, sortKey, sortDir, columns]);

  const handleSort = (key: string) => {
    const col = columns.find((c) => c.key === key);
    if (!col?.sortValue) return;
    if (sortKey === key) {
      setSortDir((d) => (d === 'asc' ? 'desc' : 'asc'));
    } else {
      setSortKey(key);
      setSortDir('asc');
    }
  };

  return (
    <table>
      <thead>
        <tr>
          {columns.map((col) => (
            <th
              key={col.key}
              onClick={() => handleSort(col.key)}
              style={{ cursor: col.sortValue ? 'pointer' : 'default', userSelect: 'none' }}
            >
              {col.label}
              {sortKey === col.key && (
                <span style={{ marginLeft: '0.25rem', fontSize: '0.6rem' }}>
                  {sortDir === 'asc' ? '\u25b2' : '\u25bc'}
                </span>
              )}
            </th>
          ))}
        </tr>
      </thead>
      <tbody>
        {sorted.map((row) => (
          <tr key={rowKey(row)} onClick={() => onRowClick?.(row)}>
            {columns.map((col) => (
              <td key={col.key}>{col.render(row)}</td>
            ))}
          </tr>
        ))}
        {sorted.length === 0 && (
          <tr style={{ cursor: 'default' }}>
            <td colSpan={columns.length} style={{ textAlign: 'center', color: 'var(--text-muted)', padding: '2rem' }}>
              No data
            </td>
          </tr>
        )}
      </tbody>
    </table>
  );
}
