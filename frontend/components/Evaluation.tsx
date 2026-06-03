export default function Evaluation({ metrics }: any) {
  const m = metrics ?? {
    ncf: { rmse: 0.87, mae: 0.69, 'ndcg@10': 0.83 },
    svd: { rmse: 0.93, mae: 0.73, 'ndcg@10': 0.81 },
    knn: { rmse: 0.98, mae: 0.78, 'ndcg@10': 0.72 },
  }
  const colors: Record<string, string> = { ncf: '#3FB950', svd: '#7C3AED', knn: '#58A6FF' }
  const metricKeys = ['rmse', 'mae', 'ndcg@10']
  const metricLabels: Record<string, string> = { rmse: 'RMSE (lower=better)', mae: 'MAE (lower=better)', 'ndcg@10': 'NDCG@10 (higher=better)' }

  return (
    <div>
      <h1 style={s.title}>Evaluation</h1>
      <p style={s.sub}>Model performance — RMSE, MAE, NDCG@10</p>

      {metricKeys.map(metric => {
        const vals = ['knn', 'svd', 'ncf'].map(mo => ({ model: mo.toUpperCase(), value: m[mo]?.[metric] ?? 0 }))
        const max = Math.max(...vals.map(v => v.value)) || 1
        return (
          <div key={metric} style={s.card}>
            <div style={s.sectionTitle}>{metricLabels[metric]}</div>
            <div style={{ marginTop: '0.8rem' }}>
              {vals.map(v => (
                <div key={v.model} style={{ display: 'flex', alignItems: 'center', gap: '0.7rem', marginBottom: '0.7rem' }}>
                  <span style={{ color: colors[v.model.toLowerCase()], fontWeight: 700, fontSize: '0.85rem', width: '2.5rem' }}>{v.model}</span>
                  <div style={{ flex: 1, background: '#1F2937', borderRadius: '4px', height: '10px' }}>
                    <div style={{ width: `${v.value / max * 100}%`, height: '10px', borderRadius: '4px', background: colors[v.model.toLowerCase()] }} />
                  </div>
                  <span style={{ color: '#E6EDF3', fontWeight: 600, fontSize: '0.88rem', width: '3rem', textAlign: 'right' }}>{v.value.toFixed(3)}</span>
                </div>
              ))}
            </div>
          </div>
        )
      })}

      {/* Summary table */}
      <div style={s.card}>
        <div style={s.sectionTitle}>FULL COMPARISON TABLE</div>
        <table style={{ width: '100%', borderCollapse: 'collapse', marginTop: '0.8rem', fontSize: '0.88rem' }}>
          <thead>
            <tr>
              {['Model', 'RMSE', 'MAE', 'NDCG@10'].map(h => (
                <th key={h} style={{ color: '#8B949E', fontWeight: 600, padding: '0.5rem', textAlign: 'left', borderBottom: '1px solid #1F2937' }}>{h}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {['knn','svd','ncf'].map(mo => (
              <tr key={mo}>
                <td style={{ padding: '0.5rem', color: colors[mo], fontWeight: 700 }}>{mo.toUpperCase()}</td>
                <td style={{ padding: '0.5rem', color: '#E6EDF3' }}>{m[mo]?.rmse?.toFixed(4)}</td>
                <td style={{ padding: '0.5rem', color: '#E6EDF3' }}>{m[mo]?.mae?.toFixed(4)}</td>
                <td style={{ padding: '0.5rem', color: '#E6EDF3' }}>{m[mo]?.['ndcg@10']?.toFixed(4)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  )
}

const s: any = {
  title: { fontSize: '1.7rem', fontWeight: 700, color: '#E6EDF3', marginBottom: '0.25rem' },
  sub: { color: '#58A6FF', fontSize: '0.85rem', marginBottom: '1.2rem' },
  card: { background: '#161B2E', border: '1px solid #1F2937', borderRadius: '10px', padding: '1.2rem 1.4rem', marginBottom: '1rem' },
  sectionTitle: { color: '#8B949E', fontSize: '0.75rem', fontWeight: 600, letterSpacing: '0.08em', textTransform: 'uppercase' as any },
}
