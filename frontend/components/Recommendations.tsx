import { useState } from 'react'
import RecList from './RecList'

export default function Recommendations({ model }: { model: string }) {
  const [uid, setUid] = useState(1)
  const [n, setN] = useState(10)
  const [recs, setRecs] = useState<any[]>([])
  const [loading, setLoading] = useState(false)

  const go = async () => {
    setLoading(true)
    try {
      const r = await fetch(`/api/recommend/${uid}?model=${model}&n=${n}`)
      const d = await r.json()
      setRecs(d.recommendations ?? [])
    } catch {}
    setLoading(false)
  }

  return (
    <div>
      <h1 style={s.title}>Recommendations</h1>
      <p style={s.sub}>Personalized top-N movie recommendations</p>
      <div style={s.controls}>
        <div>
          <label style={s.label}>User ID</label>
          <input type="number" min={1} max={943} value={uid} onChange={e => setUid(+e.target.value)} style={s.input} />
        </div>
        <div>
          <label style={s.label}>Top-N</label>
          <input type="number" min={5} max={50} step={5} value={n} onChange={e => setN(+e.target.value)} style={s.input} />
        </div>
        <button style={s.btn} onClick={go} disabled={loading}>{loading ? 'Loading…' : 'Generate'}</button>
      </div>
      <div style={s.card}>
        <RecList recs={recs} model={model} />
      </div>
    </div>
  )
}

const s: any = {
  title: { fontSize: '1.7rem', fontWeight: 700, color: '#E6EDF3', marginBottom: '0.25rem' },
  sub: { color: '#58A6FF', fontSize: '0.85rem', marginBottom: '1.2rem' },
  controls: { display: 'flex', gap: '1rem', alignItems: 'flex-end', marginBottom: '1rem', flexWrap: 'wrap' as any },
  label: { display: 'block', color: '#8B949E', fontSize: '0.78rem', marginBottom: '0.3rem' },
  input: { background: '#1A2235', border: '1px solid #30363D', color: '#E6EDF3', borderRadius: '6px', padding: '0.4rem 0.7rem', fontSize: '0.9rem', width: '100px' },
  btn: { background: '#1F6FEB', color: 'white', border: 'none', borderRadius: '6px', padding: '0.45rem 1.2rem', fontWeight: 600, cursor: 'pointer', fontSize: '0.9rem' },
  card: { background: '#161B2E', border: '1px solid #1F2937', borderRadius: '10px', padding: '1.2rem 1.3rem' },
}
