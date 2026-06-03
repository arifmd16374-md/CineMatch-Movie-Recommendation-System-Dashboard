import { useState, useEffect } from 'react'
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell } from 'recharts'
import RecList from './RecList'

export default function Dashboard({ model, stats, metrics }: any) {
  const [recs, setRecs] = useState<any[]>([])
  const [userId, setUserId] = useState(196)
  const [loading, setLoading] = useState(false)
  const [ncfHistory, setNcfHistory] = useState<number[]>([])

  const m = metrics?.[model] ?? { rmse: 0.87, mae: 0.69, 'ndcg@10': 0.83 }
  const svd = metrics?.svd ?? { rmse: 0.93, mae: 0.73 }
  const s = stats ?? { total_users: 943, total_movies: 1682, total_ratings: 100000, avg_rating: 3.53 }

  useEffect(() => {
    fetch('/api/ncf_history').then(r => r.json()).then(d => setNcfHistory(d.history ?? [])).catch(() => {})
    getRecs(196)
  }, [])

  const getRecs = async (uid: number) => {
    setLoading(true)
    try {
      const r = await fetch(`/api/recommend/${uid}?model=${model}&n=10`)
      const d = await r.json()
      setRecs(d.recommendations ?? [])
    } catch {}
    setLoading(false)
  }

  const dRmse = (m.rmse - svd.rmse).toFixed(2)
  const dMae  = (m.mae  - svd.mae).toFixed(2)

  const compData = [
    { name: 'KNN', value: metrics?.knn?.rmse ?? 0.98 },
    { name: 'SVD', value: metrics?.svd?.rmse ?? 0.93 },
    { name: 'NCF', value: metrics?.ncf?.rmse ?? 0.87 },
  ]
  const bestLoss = ncfHistory.length ? Math.min(...ncfHistory) : 0.41
  const bestEpoch = ncfHistory.length ? ncfHistory.indexOf(bestLoss) + 1 : 18

  return (
    <div>
      {/* Header */}
      <div style={s.header}>
        <div>
          <h1 style={s.title}>Dashboard</h1>
          <p style={s.sub}>MovieLens 100K &nbsp;·&nbsp; {model.toUpperCase()} model &nbsp;·&nbsp; CineMatch</p>
        </div>
        <button style={s.retrainBtn}>↗ Retrain Model</button>
      </div>

      {/* KPI Cards */}
      <div style={s.kpiRow}>
        {[
          { label: 'RMSE',    value: m.rmse.toFixed(2),  delta: `${dRmse} vs SVD`,  dc: '#3FB950' },
          { label: 'MAE',     value: m.mae.toFixed(2),   delta: `${dMae} vs SVD`,   dc: '#3FB950' },
          { label: 'NDCG@10', value: (m['ndcg@10'] ?? 0.83).toFixed(2), delta: '+2.1% this run', dc: '#3FB950' },
          { label: 'Users',   value: s.total_users.toLocaleString(), delta: `${s.total_movies.toLocaleString()} items`, dc: '#58A6FF' },
        ].map(k => (
          <div key={k.label} style={s.kpiCard}>
            <div style={s.kpiValue}>{k.value}</div>
            <div style={s.kpiLabel}>{k.label}</div>
            <div style={{ color: k.dc, fontSize: '0.78rem', marginTop: '0.25rem' }}>{k.delta}</div>
          </div>
        ))}
      </div>

      {/* Body */}
      <div style={s.body}>
        {/* Left: Recommendations */}
        <div style={s.card}>
          <div style={s.cardHeader}>
            <span style={s.sectionTitle}>TOP-10 RECOMMENDATIONS</span>
            <div style={{ display: 'flex', gap: '0.5rem', alignItems: 'center' }}>
              <input
                type="number" min={1} max={943} value={userId}
                onChange={e => setUserId(+e.target.value)}
                style={s.input}
              />
              <button style={s.btn} onClick={() => getRecs(userId)} disabled={loading}>
                {loading ? '...' : 'Get recs'}
              </button>
            </div>
          </div>
          <RecList recs={recs} model={model} />
        </div>

        {/* Right: Charts */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
          {/* Model Comparison */}
          <div style={s.card}>
            <div style={s.sectionTitle}>MODEL COMPARISON</div>
            <div style={{ marginTop: '0.8rem' }}>
              {compData.map((d, i) => {
                const colors = ['#58A6FF', '#7C3AED', '#3FB950']
                const max = 1.0
                const pct = d.value / max * 100
                return (
                  <div key={d.name} style={{ display: 'flex', alignItems: 'center', gap: '0.7rem', marginBottom: '0.7rem' }}>
                    <span style={{ color: '#8B949E', fontSize: '0.82rem', width: '2.5rem' }}>{d.name}</span>
                    <div style={{ flex: 1, background: '#1F2937', borderRadius: '4px', height: '8px' }}>
                      <div style={{ width: `${pct}%`, height: '8px', borderRadius: '4px', background: colors[i] }} />
                    </div>
                    <span style={{ color: '#E6EDF3', fontSize: '0.82rem', fontWeight: 600, width: '2.5rem', textAlign: 'right' }}>{d.value.toFixed(2)}</span>
                  </div>
                )
              })}
            </div>
          </div>

          {/* Training Loss */}
          <div style={s.card}>
            <div style={s.sectionTitle}>TRAINING LOSS (NCF)</div>
            <div style={{ marginTop: '0.5rem', height: '160px' }}>
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={ncfHistory.map((v, i) => ({ epoch: i + 1, loss: v }))}>
                  <XAxis dataKey="epoch" tick={{ fontSize: 10, fill: '#484F58' }}
                    tickLine={false} axisLine={false}
                    ticks={[1, ncfHistory.length || 20]}
                    tickFormatter={v => v === 1 ? 'Epoch 1' : `Epoch ${v}`} />
                  <YAxis hide />
                  <Tooltip
                    contentStyle={{ background: '#161B2E', border: '1px solid #1F2937', borderRadius: '6px', fontSize: '0.78rem' }}
                    formatter={(v: any) => [v.toFixed(4), 'Loss']}
                    labelFormatter={(l: any) => `Epoch ${l}`}
                  />
                  <Bar dataKey="loss" radius={[2, 2, 0, 0]}>
                    {(ncfHistory.length ? ncfHistory : Array(20).fill(0)).map((_, i) => (
                      <Cell key={i} fill={i >= (ncfHistory.length || 20) - 3 ? '#3FB950' : '#58A6FF'} />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginTop: '0.4rem' }}>
              <span style={{ color: '#58A6FF', fontSize: '0.8rem' }}>
                Best loss: <b>{bestLoss.toFixed(2)}</b> @ epoch {bestEpoch}
              </span>
              <button style={{ ...s.btn, padding: '0.3rem 0.8rem', fontSize: '0.78rem' }}>Optimise ↗</button>
            </div>
          </div>

          {/* Genre Coverage */}
          <div style={s.card}>
            <div style={s.sectionTitle}>GENRE COVERAGE — RECOMMENDED VS DATASET</div>
            <div style={{ marginTop: '0.8rem' }}>
              {[
                { genre: 'Action',   rec: 28, ds: 22 },
                { genre: 'Comedy',   rec: 15, ds: 18 },
                { genre: 'Drama',    rec: 34, ds: 30 },
                { genre: 'Sci-Fi',   rec: 16, ds: 12 },
                { genre: 'Thriller', rec: 18, ds: 14 },
              ].map(g => (
                <div key={g.genre} style={{ display: 'flex', alignItems: 'center', marginBottom: '0.65rem', gap: '0.6rem' }}>
                  <span style={{ color: '#C9D1D9', fontSize: '0.83rem', width: '5rem' }}>{g.genre}</span>
                  <div style={{ flex: 1, position: 'relative', height: '12px' }}>
                    <div style={{ position: 'absolute', left: 0, top: 1, height: '10px', width: `${g.ds * 2}%`, background: '#3FB950', opacity: 0.65, borderRadius: '3px' }} />
                    <div style={{ position: 'absolute', left: 0, top: 3, height: '6px', width: `${g.rec * 2}%`, background: '#58A6FF', opacity: 0.85, borderRadius: '3px' }} />
                  </div>
                  <span style={{ color: '#8B949E', fontSize: '0.72rem', width: '13rem', textAlign: 'right' }}>{g.rec}% rec · {g.ds}% dataset</span>
                </div>
              ))}
              <div style={{ display: 'flex', gap: '1.2rem', marginTop: '0.5rem' }}>
                {[{ color: '#58A6FF', label: 'Recommended' }, { color: '#3FB950', label: 'Dataset avg' }].map(l => (
                  <div key={l.label} style={{ display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
                    <div style={{ width: 14, height: 8, borderRadius: 2, background: l.color, opacity: 0.85 }} />
                    <span style={{ color: '#8B949E', fontSize: '0.75rem' }}>{l.label}</span>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}

const s: any = {
  header: { display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', paddingBottom: '1rem', borderBottom: '1px solid #1F2937', marginBottom: '1.2rem' },
  title: { fontSize: '1.7rem', fontWeight: 700, color: '#E6EDF3' },
  sub: { color: '#58A6FF', fontSize: '0.85rem', marginTop: '0.2rem' },
  retrainBtn: { background: '#1F2D4A', border: '1px solid #388BFD', color: '#58A6FF', padding: '0.4rem 1rem', borderRadius: '6px', fontSize: '0.85rem', fontWeight: 600, cursor: 'pointer' },
  kpiRow: { display: 'flex', gap: '1rem', marginBottom: '1.3rem', flexWrap: 'wrap' as any },
  kpiCard: { flex: 1, minWidth: '150px', background: '#161B2E', border: '1px solid #1F2937', borderRadius: '10px', padding: '1.1rem 1.3rem' },
  kpiValue: { fontSize: '2rem', fontWeight: 700, color: '#E6EDF3', lineHeight: 1 },
  kpiLabel: { color: '#8B949E', fontSize: '0.8rem', marginTop: '0.25rem' },
  body: { display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1rem' },
  card: { background: '#161B2E', border: '1px solid #1F2937', borderRadius: '10px', padding: '1.2rem 1.3rem' },
  cardHeader: { display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.8rem', flexWrap: 'wrap' as any, gap: '0.5rem' },
  sectionTitle: { color: '#8B949E', fontSize: '0.75rem', fontWeight: 600, letterSpacing: '0.08em', textTransform: 'uppercase' as any },
  input: { background: '#0D1117', border: '1px solid #30363D', color: '#E6EDF3', borderRadius: '6px', padding: '0.35rem 0.6rem', width: '75px', fontSize: '0.85rem' },
  btn: { background: '#1F6FEB', color: 'white', border: 'none', borderRadius: '6px', padding: '0.38rem 0.9rem', fontWeight: 600, cursor: 'pointer', fontSize: '0.85rem' },
}
