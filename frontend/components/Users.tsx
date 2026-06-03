import { useState } from 'react'

export default function Users() {
  const [uid, setUid] = useState(1)
  const [recs, setRecs] = useState<any[]>([])
  const [loading, setLoading] = useState(false)

  const go = async () => {
    setLoading(true)
    try {
      const r = await fetch(`/api/recommend/${uid}?model=ncf&n=5`)
      const d = await r.json()
      setRecs(d.recommendations ?? [])
    } catch {}
    setLoading(false)
  }

  const kpis = [
    { label: 'Total Users', value: '943' },
    { label: 'Avg Ratings/User', value: '106' },
    { label: 'Most Active', value: 'User 405' },
    { label: 'Rating Scale', value: '1 – 5 ★' },
  ]

  return (
    <div>
      <h1 style={s.title}>Users</h1>
      <p style={s.sub}>User activity and rating behaviour</p>
      <div style={s.kpiRow}>
        {kpis.map(k => (
          <div key={k.label} style={s.kpiCard}>
            <div style={s.kpiValue}>{k.value}</div>
            <div style={s.kpiLabel}>{k.label}</div>
          </div>
        ))}
      </div>
      <div style={s.card}>
        <div style={s.cardHeader}>
          <span style={s.sectionTitle}>GET TOP RECOMMENDATIONS FOR A USER</span>
          <div style={{ display: 'flex', gap: '0.5rem' }}>
            <input type="number" min={1} max={943} value={uid}
              onChange={e => setUid(+e.target.value)} style={s.input} />
            <button style={s.btn} onClick={go}>{loading ? '…' : 'Lookup'}</button>
          </div>
        </div>
        {recs.map((rec, i) => (
          <div key={rec.movie_id} style={s.row}>
            <span style={{ color: '#484F58', fontSize: '0.78rem', width: '1.8rem' }}>#{i+1}</span>
            <div style={{ flex: 1, color: '#E6EDF3', fontSize: '0.88rem' }}>{rec.title}</div>
            <div style={{ color: '#3ECFCF', fontWeight: 700 }}>{rec.predicted_rating.toFixed(2)}</div>
          </div>
        ))}
        {recs.length === 0 && <p style={{ color: '#484F58', textAlign: 'center', fontSize: '0.85rem' }}>Enter a User ID and click Lookup</p>}
      </div>
    </div>
  )
}

const s: any = {
  title: { fontSize: '1.7rem', fontWeight: 700, color: '#E6EDF3', marginBottom: '0.25rem' },
  sub: { color: '#58A6FF', fontSize: '0.85rem', marginBottom: '1.2rem' },
  kpiRow: { display: 'flex', gap: '1rem', marginBottom: '1.2rem', flexWrap: 'wrap' as any },
  kpiCard: { flex: 1, minWidth: '140px', background: '#161B2E', border: '1px solid #1F2937', borderRadius: '10px', padding: '1rem 1.2rem' },
  kpiValue: { fontSize: '1.8rem', fontWeight: 700, color: '#E6EDF3' },
  kpiLabel: { color: '#8B949E', fontSize: '0.78rem', marginTop: '0.2rem' },
  card: { background: '#161B2E', border: '1px solid #1F2937', borderRadius: '10px', padding: '1.2rem 1.3rem' },
  cardHeader: { display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1rem', flexWrap: 'wrap' as any, gap: '0.5rem' },
  sectionTitle: { color: '#8B949E', fontSize: '0.75rem', fontWeight: 600, letterSpacing: '0.08em', textTransform: 'uppercase' as any },
  input: { background: '#0D1117', border: '1px solid #30363D', color: '#E6EDF3', borderRadius: '6px', padding: '0.35rem 0.6rem', width: '80px', fontSize: '0.85rem' },
  btn: { background: '#1F6FEB', color: 'white', border: 'none', borderRadius: '6px', padding: '0.38rem 0.9rem', fontWeight: 600, cursor: 'pointer' },
  row: { display: 'flex', alignItems: 'center', gap: '0.8rem', padding: '0.55rem 0.3rem', borderBottom: '1px solid #1F2937' },
}
