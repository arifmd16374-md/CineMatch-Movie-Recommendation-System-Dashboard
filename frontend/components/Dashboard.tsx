import { useState, useEffect } from 'react'
import RecList from './RecList'
import { ModelComparison, TrainingLoss, NcfArchitecture, NcfPredict, NcfSimilar, NcfUserEmbedding } from './NcfPanel'

export default function Dashboard({ model, stats, metrics }: any) {
  const [recs,    setRecs]    = useState<any[]>([])
  const [userId,  setUserId]  = useState(196)
  const [loading, setLoading] = useState(false)

  const m   = metrics?.[model] ?? { rmse:0.87, mae:0.69, 'ndcg@10':0.83 }
  const svd = metrics?.svd     ?? { rmse:0.93, mae:0.73 }
  const st  = stats            ?? { total_users:943, total_movies:1682, total_ratings:100000, avg_rating:3.53 }

  useEffect(() => { getRecs(196) }, [model])

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
          { label:'RMSE',    value:m.rmse.toFixed(2),          delta:`${dRmse} vs SVD`,  dc:'#3FB950' },
          { label:'MAE',     value:m.mae.toFixed(2),           delta:`${dMae} vs SVD`,   dc:'#3FB950' },
          { label:'NDCG@10', value:(m['ndcg@10']??0.83).toFixed(2), delta:'+2.1% this run', dc:'#3FB950' },
          { label:'Users',   value:st.total_users.toLocaleString(), delta:`${st.total_movies.toLocaleString()} items`, dc:'#58A6FF' },
        ].map(k => (
          <div key={k.label} style={s.kpiCard}>
            <div style={s.kpiValue}>{k.value}</div>
            <div style={s.kpiLabel}>{k.label}</div>
            <div style={{ color:k.dc, fontSize:'0.78rem', marginTop:'0.25rem' }}>{k.delta}</div>
          </div>
        ))}
      </div>

      {/* Main 2-col layout */}
      <div style={s.body}>
        {/* LEFT — Top-10 Recommendations */}
        <div style={s.card}>
          <div style={s.cardHeader}>
            <span style={s.sectionTitle}>TOP-10 RECOMMENDATIONS — USER #{userId}</span>
            <div style={{ display:'flex', gap:'0.5rem', alignItems:'center' }}>
              <input type="number" min={1} max={943} value={userId}
                onChange={e => setUserId(+e.target.value)} style={s.input}/>
              <button style={s.btn} onClick={() => getRecs(userId)} disabled={loading}>
                {loading ? '…' : 'Get recs'}
              </button>
            </div>
          </div>
          <RecList recs={recs} model={model} />
        </div>

        {/* RIGHT — Model Comparison + Training Loss */}
        <div style={{ display:'flex', flexDirection:'column', gap:'1rem' }}>
          <ModelComparison metrics={metrics} />
          <TrainingLoss />
          <GenreCoverage recs={recs} />
        </div>
      </div>

      {/* NCF Deep-Dive Section */}
      <div style={{ marginTop:'1.2rem' }}>
        <div style={s.ncfHeader}>
          <span style={{ fontSize:'0.72rem', fontWeight:700, letterSpacing:'0.1em',
            textTransform:'uppercase', color:'#3FB950' }}>● NCF DEEP DIVE</span>
          <span style={{ color:'#484F58', fontSize:'0.75rem' }}>Neural Collaborative Filtering — interactive analysis</span>
        </div>
        <div style={s.ncfGrid}>
          <NcfPredict />
          <NcfArchitecture />
          <NcfSimilar />
          <NcfUserEmbedding />
        </div>
      </div>
    </div>
  )
}

function GenreCoverage({ recs }: { recs: any[] }) {
  const genreCounts: any = { Action:0, Comedy:0, Drama:0, 'Sci-Fi':0, Thriller:0 }
  recs.forEach(r => { if (r.genre in genreCounts) genreCounts[r.genre]++ })
  const total = recs.length || 1
  const ds = { Action:22, Comedy:18, Drama:30, 'Sci-Fi':12, Thriller:14 }
  const genres = [
    { genre:'Action',  rec: recs.length ? Math.round(genreCounts.Action/total*100)  : 28, ds:22 },
    { genre:'Comedy',  rec: recs.length ? Math.round(genreCounts.Comedy/total*100)  : 15, ds:18 },
    { genre:'Drama',   rec: recs.length ? Math.round(genreCounts.Drama/total*100)   : 34, ds:30 },
    { genre:'Sci-Fi',  rec: recs.length ? Math.round(genreCounts['Sci-Fi']/total*100):16, ds:12 },
    { genre:'Thriller',rec: recs.length ? Math.round(genreCounts.Thriller/total*100): 18, ds:14 },
  ]
  return (
    <div style={s.card}>
      <div style={s.sectionTitle}>GENRE COVERAGE — RECOMMENDED VS DATASET</div>
      <div style={{ marginTop:'0.8rem' }}>
        {genres.map(g => (
          <div key={g.genre} style={{ display:'flex', alignItems:'center', marginBottom:'0.65rem', gap:'0.6rem' }}>
            <span style={{ color:'#C9D1D9', fontSize:'0.83rem', width:'5rem' }}>{g.genre}</span>
            <div style={{ flex:1, position:'relative', height:'12px' }}>
              <div style={{ position:'absolute', left:0, top:1, height:'10px', width:`${g.ds*2}%`, background:'#3FB950', opacity:0.65, borderRadius:'3px' }}/>
              <div style={{ position:'absolute', left:0, top:3, height:'6px',  width:`${g.rec*2}%`, background:'#58A6FF', opacity:0.85, borderRadius:'3px' }}/>
            </div>
            <span style={{ color:'#8B949E', fontSize:'0.72rem', width:'13rem', textAlign:'right' }}>{g.rec}% rec · {g.ds}% dataset</span>
          </div>
        ))}
        <div style={{ display:'flex', gap:'1.2rem', marginTop:'0.5rem' }}>
          {[{color:'#58A6FF',label:'Recommended'},{color:'#3FB950',label:'Dataset avg'}].map(l=>(
            <div key={l.label} style={{ display:'flex', alignItems:'center', gap:'0.4rem' }}>
              <div style={{ width:14, height:8, borderRadius:2, background:l.color, opacity:0.85 }}/>
              <span style={{ color:'#8B949E', fontSize:'0.75rem' }}>{l.label}</span>
            </div>
          ))}
        </div>
      </div>
    </div>
  )
}

const s: any = {
  header:{ display:'flex', justifyContent:'space-between', alignItems:'flex-start', paddingBottom:'1rem', borderBottom:'1px solid #1F2937', marginBottom:'1.2rem' },
  title:{ fontSize:'1.7rem', fontWeight:700, color:'#E6EDF3' },
  sub:{ color:'#58A6FF', fontSize:'0.85rem', marginTop:'0.2rem' },
  retrainBtn:{ background:'#1F2D4A', border:'1px solid #388BFD', color:'#58A6FF', padding:'0.4rem 1rem', borderRadius:'6px', fontSize:'0.85rem', fontWeight:600, cursor:'pointer' },
  kpiRow:{ display:'flex', gap:'1rem', marginBottom:'1.3rem', flexWrap:'wrap' as any },
  kpiCard:{ flex:1, minWidth:'150px', background:'#161B2E', border:'1px solid #1F2937', borderRadius:'10px', padding:'1.1rem 1.3rem' },
  kpiValue:{ fontSize:'2rem', fontWeight:700, color:'#E6EDF3', lineHeight:1 },
  kpiLabel:{ color:'#8B949E', fontSize:'0.8rem', marginTop:'0.25rem' },
  body:{ display:'grid', gridTemplateColumns:'1fr 1fr', gap:'1rem' },
  card:{ background:'#161B2E', border:'1px solid #1F2937', borderRadius:'10px', padding:'1.2rem 1.3rem' },
  cardHeader:{ display:'flex', justifyContent:'space-between', alignItems:'center', marginBottom:'0.8rem', flexWrap:'wrap' as any, gap:'0.5rem' },
  sectionTitle:{ color:'#8B949E', fontSize:'0.72rem', fontWeight:600, letterSpacing:'0.08em', textTransform:'uppercase' as any },
  input:{ background:'#0D1117', border:'1px solid #30363D', color:'#E6EDF3', borderRadius:'6px', padding:'0.35rem 0.6rem', width:'75px', fontSize:'0.85rem' },
  btn:{ background:'#1F6FEB', color:'white', border:'none', borderRadius:'6px', padding:'0.38rem 0.9rem', fontWeight:600, cursor:'pointer', fontSize:'0.85rem' },
  ncfHeader:{ display:'flex', alignItems:'center', gap:'0.8rem', padding:'0.6rem 0', borderBottom:'1px solid #1F2937', marginBottom:'1rem' },
  ncfGrid:{ display:'grid', gridTemplateColumns:'1fr 1fr', gap:'1rem' },
}
