import { useState, useEffect } from 'react'
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell, RadarChart, Radar, PolarGrid, PolarAngleAxis } from 'recharts'

// ── Model Comparison bars ────────────────────────────────────────────────────
export function ModelComparison({ metrics }: any) {
  const [activeMetric, setActiveMetric] = useState<'rmse'|'mae'|'ndcg@10'>('rmse')
  const m = metrics ?? { ncf:{rmse:0.87,mae:0.69,'ndcg@10':0.83}, svd:{rmse:0.93,mae:0.73,'ndcg@10':0.81}, knn:{rmse:0.98,mae:0.78,'ndcg@10':0.72} }
  const colors: any = { knn:'#58A6FF', svd:'#7C3AED', ncf:'#3FB950' }
  const vals = ['knn','svd','ncf'].map(mo => ({ name:mo.toUpperCase(), value: m[mo]?.[activeMetric] ?? 0 }))
  const maxV = Math.max(...vals.map(v=>v.value)) || 1
  return (
    <div style={s.card}>
      <div style={s.sectionTitle}>MODEL COMPARISON</div>
      <div style={{ display:'flex', gap:'0.4rem', margin:'0.7rem 0 0.8rem' }}>
        {(['rmse','mae','ndcg@10'] as const).map(mk => (
          <button key={mk} onClick={()=>setActiveMetric(mk)}
            style={{ ...s.tabBtn, ...(activeMetric===mk ? s.tabActive : {}) }}>
            {mk.toUpperCase()}
          </button>
        ))}
      </div>
      {vals.map(v => (
        <div key={v.name} style={{ display:'flex', alignItems:'center', gap:'0.7rem', marginBottom:'0.65rem' }}>
          <span style={{ color:'#8B949E', fontSize:'0.82rem', width:'2.5rem' }}>{v.name}</span>
          <div style={{ flex:1, background:'#1F2937', borderRadius:'4px', height:'8px' }}>
            <div style={{ width:`${v.value/maxV*100}%`, height:'8px', borderRadius:'4px', background:colors[v.name.toLowerCase()],
              transition:'width 0.4s ease' }} />
          </div>
          <span style={{ color:'#E6EDF3', fontSize:'0.82rem', fontWeight:600, width:'2.5rem', textAlign:'right' }}>{v.value.toFixed(2)}</span>
        </div>
      ))}
    </div>
  )
}

// ── NCF Training Loss chart with epoch drill-down ────────────────────────────
export function TrainingLoss() {
  const [history, setHistory] = useState<number[]>([])
  const [selectedEpoch, setSelectedEpoch] = useState<any>(null)
  const [loadingEpoch, setLoadingEpoch] = useState(false)

  useEffect(() => {
    fetch('/api/ncf_history').then(r=>r.json()).then(d=>setHistory(d.history??[])).catch(()=>{})
  }, [])

  const bestLoss  = history.length ? Math.min(...history) : 0.48
  const bestEpoch = history.length ? history.indexOf(bestLoss)+1 : 18
  const n = history.length || 20

  const clickBar = async (data: any) => {
    if (!data?.activePayload?.[0]) return
    const epoch = data.activePayload[0].payload.epoch
    setLoadingEpoch(true)
    try {
      const r = await fetch(`/api/ncf_epoch_detail/${epoch}`)
      setSelectedEpoch(await r.json())
    } catch {}
    setLoadingEpoch(false)
  }

  return (
    <div style={s.card}>
      <div style={s.sectionTitle}>TRAINING LOSS (NCF) — click bar for details</div>
      <div style={{ marginTop:'0.5rem', height:'160px' }}>
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={(history.length?history:Array(20).fill(0.6)).map((v,i)=>({epoch:i+1,loss:v}))}
            onClick={clickBar} style={{cursor:'pointer'}}>
            <XAxis dataKey="epoch" tick={{fontSize:10,fill:'#484F58'}} tickLine={false} axisLine={false}
              ticks={[1,n]} tickFormatter={v=>v===1?'Epoch 1':`Epoch ${v}`}/>
            <YAxis hide />
            <Tooltip contentStyle={{background:'#161B2E',border:'1px solid #1F2937',borderRadius:'6px',fontSize:'0.78rem'}}
              formatter={(v:any)=>[v.toFixed(4),'Loss']} labelFormatter={(l:any)=>`Epoch ${l} — click for detail`}/>
            <Bar dataKey="loss" radius={[2,2,0,0]}>
              {(history.length?history:Array(20).fill(0)).map((_,i)=>(
                <Cell key={i}
                  fill={selectedEpoch?.epoch===i+1 ? '#F59E0B' : i>=n-3 ? '#3FB950' : '#58A6FF'}/>
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>
      <div style={{display:'flex',justifyContent:'space-between',alignItems:'center',marginTop:'0.4rem'}}>
        <span style={{color:'#58A6FF',fontSize:'0.8rem'}}>
          Best loss: <b>{bestLoss.toFixed(2)}</b> @ epoch {bestEpoch}
        </span>
        <span style={{color:'#484F58',fontSize:'0.72rem'}}>Click bar to inspect epoch</span>
      </div>
      {/* Epoch detail popup */}
      {selectedEpoch && (
        <div style={{marginTop:'0.7rem',background:'#0D1117',borderRadius:'8px',padding:'0.8rem',border:'1px solid #2d3f54'}}>
          <div style={{display:'flex',justifyContent:'space-between',alignItems:'center',marginBottom:'0.5rem'}}>
            <span style={{color:'#E6EDF3',fontWeight:600,fontSize:'0.85rem'}}>Epoch {selectedEpoch.epoch} Detail</span>
            {selectedEpoch.improved && <span style={{background:'rgba(63,185,80,0.15)',color:'#3FB950',fontSize:'0.7rem',padding:'0.1rem 0.5rem',borderRadius:'4px'}}>Best checkpoint ✓</span>}
            <button onClick={()=>setSelectedEpoch(null)} style={{background:'transparent',border:'none',color:'#484F58',cursor:'pointer',fontSize:'1rem'}}>✕</button>
          </div>
          <div style={{display:'grid',gridTemplateColumns:'1fr 1fr',gap:'0.5rem'}}>
            {[
              {label:'Train Loss', value: selectedEpoch.train_loss, color:'#58A6FF'},
              {label:'Val Loss',   value: selectedEpoch.val_loss,   color:'#7C3AED'},
              {label:'Train RMSE',value: selectedEpoch.train_rmse, color:'#3FB950'},
              {label:'Val RMSE',  value: selectedEpoch.val_rmse,   color:'#3ECFCF'},
            ].map(item=>(
              <div key={item.label} style={{background:'#161B2E',borderRadius:'6px',padding:'0.4rem 0.6rem',border:'1px solid #1F2937'}}>
                <div style={{color:'#8B949E',fontSize:'0.7rem'}}>{item.label}</div>
                <div style={{color:item.color,fontWeight:700,fontSize:'1rem'}}>{item.value}</div>
              </div>
            ))}
          </div>
          <div style={{color:'#484F58',fontSize:'0.72rem',marginTop:'0.4rem'}}>LR: {selectedEpoch.lr} · Optimizer: Adam · Loss: MSELoss</div>
        </div>
      )}
    </div>
  )
}

// ── NCF Architecture panel ────────────────────────────────────────────────────
export function NcfArchitecture() {
  const [arch, setArch] = useState<any>(null)
  useEffect(()=>{
    fetch('/api/ncf_architecture').then(r=>r.json()).then(setArch).catch(()=>{})
  },[])
  const a = arch ?? {embedding_dim:64,hidden_layers:[256,128,64],dropout_rate:0.3,
    learning_rate:0.001,batch_size:256,optimizer:'Adam',loss_fn:'MSELoss',
    total_params:185600,n_users:943,n_movies:1682}
  return (
    <div style={s.card}>
      <div style={s.sectionTitle}>NCF ARCHITECTURE</div>
      <div style={{marginTop:'0.7rem'}}>
        {/* Layer diagram */}
        <div style={{display:'flex',alignItems:'center',gap:'0.3rem',marginBottom:'0.8rem',overflowX:'auto',padding:'0.3rem 0'}}>
          {[
            {label:`User Emb\n${a.n_users}×${a.embedding_dim}`,color:'#58A6FF'},
            {label:`Movie Emb\n${a.n_movies}×${a.embedding_dim}`,color:'#7C3AED'},
            {label:`Concat\n${a.embedding_dim*2}`,color:'#484F58'},
            ...(a.hidden_layers??[256,128,64]).map((h:number,i:number)=>({label:`Dense\n${h}`,color:i===0?'#3FB950':i===1?'#3ECFCF':'#F59E0B'})),
            {label:'Output\n1',color:'#EF4444'},
          ].map((layer,i,arr)=>(
            <div key={i} style={{display:'flex',alignItems:'center',gap:'0.3rem'}}>
              <div style={{background:'rgba(0,0,0,0.3)',border:`1px solid ${layer.color}`,borderRadius:'6px',
                padding:'0.3rem 0.5rem',textAlign:'center',minWidth:'60px',flexShrink:0}}>
                {layer.label.split('\n').map((t,j)=>(
                  <div key={j} style={{color:j===0?layer.color:'#8B949E',fontSize:j===0?'0.7rem':'0.65rem',fontWeight:j===0?600:400,whiteSpace:'nowrap'}}>{t}</div>
                ))}
              </div>
              {i<arr.length-1 && <span style={{color:'#484F58',fontSize:'0.7rem',flexShrink:0}}>→</span>}
            </div>
          ))}
        </div>
        {/* Params */}
        <div style={{display:'grid',gridTemplateColumns:'1fr 1fr',gap:'0.4rem'}}>
          {[
            {k:'Embedding Dim', v:`${a.embedding_dim}`},
            {k:'Hidden Layers', v:(a.hidden_layers??[256,128,64]).join(' → ')},
            {k:'Dropout', v:`${a.dropout_rate}`},
            {k:'Learning Rate', v:`${a.learning_rate}`},
            {k:'Batch Size', v:`${a.batch_size}`},
            {k:'Optimizer', v:a.optimizer??'Adam'},
            {k:'Loss', v:a.loss_fn??'MSELoss'},
            {k:'Total Params', v:(a.total_params??185600).toLocaleString()},
          ].map(item=>(
            <div key={item.k} style={{display:'flex',justifyContent:'space-between',padding:'0.3rem 0',borderBottom:'1px solid #1F2937'}}>
              <span style={{color:'#8B949E',fontSize:'0.78rem'}}>{item.k}</span>
              <span style={{color:'#E6EDF3',fontSize:'0.78rem',fontWeight:600}}>{item.v}</span>
            </div>
          ))}
        </div>
      </div>
    </div>
  )
}

// ── NCF Predict — all models side by side ─────────────────────────────────────
export function NcfPredict() {
  const [uid, setUid] = useState(196)
  const [mid, setMid] = useState(50)
  const [result, setResult] = useState<any>(null)
  const [loading, setLoading] = useState(false)

  const run = async () => {
    setLoading(true)
    try {
      const r = await fetch('/api/predict_all_models', {
        method:'POST', headers:{'Content-Type':'application/json'},
        body: JSON.stringify({user_id:uid, movie_id:mid})
      })
      setResult(await r.json())
    } catch {}
    setLoading(false)
  }

  const colors: any = {ncf:'#3FB950',svd:'#7C3AED',knn:'#58A6FF'}

  return (
    <div style={s.card}>
      <div style={s.sectionTitle}>NCF PREDICTION — ALL MODELS COMPARED</div>
      <div style={{display:'flex',gap:'0.5rem',margin:'0.7rem 0',flexWrap:'wrap',alignItems:'flex-end'}}>
        <div>
          <div style={s.inputLabel}>User ID</div>
          <input type="number" min={1} max={943} value={uid} onChange={e=>setUid(+e.target.value)} style={s.input}/>
        </div>
        <div>
          <div style={s.inputLabel}>Movie ID</div>
          <input type="number" min={1} max={1682} value={mid} onChange={e=>setMid(+e.target.value)} style={s.input}/>
        </div>
        <button style={s.btn} onClick={run} disabled={loading}>{loading?'…':'Predict'}</button>
      </div>
      {result && (
        <>
          <div style={{color:'#8B949E',fontSize:'0.78rem',marginBottom:'0.6rem'}}>
            Movie: <span style={{color:'#E6EDF3',fontWeight:600}}>{result.title}</span>
          </div>
          <div style={{display:'flex',gap:'0.5rem'}}>
            {Object.entries(result.predictions).map(([model,score]:any)=>(
              <div key={model} style={{flex:1,background:'#0D1117',border:`1px solid ${colors[model]}22`,borderRadius:'8px',padding:'0.8rem',textAlign:'center',position:'relative'}}>
                <div style={{color:colors[model],fontSize:'0.72rem',fontWeight:700,textTransform:'uppercase',letterSpacing:'0.06em',marginBottom:'0.3rem'}}>{model}</div>
                <div style={{fontSize:'1.8rem',fontWeight:800,color:'#E6EDF3',lineHeight:1}}>{score.toFixed(2)}</div>
                <div style={{color:'#484F58',fontSize:'0.68rem',marginTop:'0.2rem'}}>/ 5.00</div>
                {model==='ncf' && (
                  <div style={{position:'absolute',top:4,right:4,background:'rgba(63,185,80,0.15)',color:'#3FB950',fontSize:'0.6rem',padding:'0.1rem 0.3rem',borderRadius:'3px'}}>Best</div>
                )}
              </div>
            ))}
          </div>
          {/* Visual bar */}
          <div style={{marginTop:'0.8rem'}}>
            {Object.entries(result.predictions).map(([model,score]:any)=>(
              <div key={model} style={{display:'flex',alignItems:'center',gap:'0.6rem',marginBottom:'0.4rem'}}>
                <span style={{color:colors[model],fontSize:'0.75rem',width:'2.5rem',fontWeight:600}}>{model.toUpperCase()}</span>
                <div style={{flex:1,background:'#1F2937',borderRadius:'3px',height:'6px'}}>
                  <div style={{width:`${score/5*100}%`,height:'6px',borderRadius:'3px',background:colors[model],transition:'width 0.5s ease'}}/>
                </div>
                <div style={{position:'relative',width:'6rem',height:'6px',background:'#1F2937',borderRadius:'3px'}}>
                  <div style={{width:`${score/5*100}%`,height:'6px',borderRadius:'3px',
                    background:`linear-gradient(90deg,${colors[model]}88,${colors[model]})`,transition:'width 0.5s ease'}}/>
                </div>
                <span style={{color:'#8B949E',fontSize:'0.75rem',width:'2rem',textAlign:'right'}}>{score.toFixed(2)}</span>
              </div>
            ))}
            <div style={{borderTop:'1px solid #1F2937',marginTop:'0.4rem',paddingTop:'0.3rem',color:'#484F58',fontSize:'0.7rem',textAlign:'right'}}>
              Dataset mean: 3.53
            </div>
          </div>
        </>
      )}
    </div>
  )
}

// ── NCF Similar Movies (embedding cosine similarity) ─────────────────────────
export function NcfSimilar() {
  const [mid, setMid] = useState(50)
  const [result, setResult] = useState<any>(null)
  const [loading, setLoading] = useState(false)

  const run = async () => {
    setLoading(true)
    try {
      const r = await fetch(`/api/ncf_similar/${mid}?n=6`)
      setResult(await r.json())
    } catch {}
    setLoading(false)
  }

  const GENRE_COLORS: any = {
    Drama:{bg:'rgba(124,58,237,0.15)',color:'#A78BFA'},
    Thriller:{bg:'rgba(210,153,63,0.15)',color:'#D2993F'},
    'Sci-Fi':{bg:'rgba(88,166,255,0.12)',color:'#58A6FF'},
    Comedy:{bg:'rgba(63,185,80,0.15)',color:'#3FB950'},
    Action:{bg:'rgba(63,185,80,0.15)',color:'#3FB950'},
    Horror:{bg:'rgba(239,68,68,0.15)',color:'#EF4444'},
    Animation:{bg:'rgba(249,115,22,0.15)',color:'#F97316'},
  }

  return (
    <div style={s.card}>
      <div style={s.sectionTitle}>NCF EMBEDDING SIMILARITY — find similar movies</div>
      <div style={{display:'flex',gap:'0.5rem',margin:'0.7rem 0',alignItems:'flex-end'}}>
        <div>
          <div style={s.inputLabel}>Movie ID (1–1682)</div>
          <input type="number" min={1} max={1682} value={mid} onChange={e=>setMid(+e.target.value)} style={s.input}/>
        </div>
        <button style={s.btn} onClick={run} disabled={loading}>{loading?'…':'Find Similar'}</button>
      </div>
      {result && (
        <>
          <div style={{color:'#8B949E',fontSize:'0.78rem',marginBottom:'0.7rem'}}>
            Similar to: <span style={{color:'#E6EDF3',fontWeight:600}}>{result.title}</span>
            <span style={{color:'#484F58',marginLeft:'0.5rem',fontSize:'0.72rem'}}>via NCF movie embeddings (cosine similarity)</span>
          </div>
          {result.similar.map((m:any,i:number)=>{
            const gc = GENRE_COLORS[m.genre]??{bg:'rgba(88,166,255,0.1)',color:'#58A6FF'}
            const simPct = Math.round((m.similarity+1)/2*100)
            return (
              <div key={m.movie_id} style={{display:'flex',alignItems:'center',gap:'0.7rem',padding:'0.5rem 0',borderBottom:'1px solid #1F2937'}}>
                <span style={{color:'#484F58',fontSize:'0.72rem',width:'1.2rem'}}>{i+1}</span>
                <div style={{flex:1,minWidth:0}}>
                  <div style={{color:'#E6EDF3',fontSize:'0.84rem',fontWeight:500,overflow:'hidden',textOverflow:'ellipsis',whiteSpace:'nowrap'}}>{m.title}</div>
                  <span style={{...gc,fontSize:'0.66rem',borderRadius:'3px',padding:'0.05rem 0.35rem',display:'inline-block',marginTop:'2px'}}>{m.genre}</span>
                </div>
                <div style={{flexShrink:0,textAlign:'right'}}>
                  <div style={{color:'#3ECFCF',fontWeight:700,fontSize:'0.9rem'}}>{m.similarity.toFixed(3)}</div>
                  <div style={{width:'60px',height:'4px',background:'#1F2937',borderRadius:'2px',marginTop:'3px'}}>
                    <div style={{width:`${simPct}%`,height:'4px',background:'#3ECFCF',borderRadius:'2px'}}/>
                  </div>
                </div>
              </div>
            )
          })}
        </>
      )}
    </div>
  )
}

// ── NCF User Embedding ────────────────────────────────────────────────────────
export function NcfUserEmbedding() {
  const [uid, setUid] = useState(196)
  const [data, setData] = useState<any>(null)

  const run = async () => {
    try {
      const r = await fetch(`/api/ncf_user_embedding/${uid}`)
      setData(await r.json())
    } catch {}
  }

  return (
    <div style={s.card}>
      <div style={s.sectionTitle}>NCF USER LATENT FACTORS — top-10 embedding dimensions</div>
      <div style={{display:'flex',gap:'0.5rem',margin:'0.7rem 0',alignItems:'flex-end'}}>
        <div>
          <div style={s.inputLabel}>User ID</div>
          <input type="number" min={1} max={943} value={uid} onChange={e=>setUid(+e.target.value)} style={s.input}/>
        </div>
        <button style={s.btn} onClick={run}>Visualise</button>
      </div>
      {data && (
        <div style={{height:'140px'}}>
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={data.top_dimensions.map((d:any)=>({name:`D${d.dim}`,value:Math.abs(d.value),raw:d.value}))}>
              <XAxis dataKey="name" tick={{fontSize:9,fill:'#484F58'}} tickLine={false} axisLine={false}/>
              <YAxis hide/>
              <Tooltip contentStyle={{background:'#161B2E',border:'1px solid #1F2937',borderRadius:'6px',fontSize:'0.75rem'}}
                formatter={(_:any,__:any,p:any)=>[p.payload.raw.toFixed(4),'Value']}/>
              <Bar dataKey="value" radius={[2,2,0,0]}>
                {data.top_dimensions.map((d:any,i:number)=>(
                  <Cell key={i} fill={d.value>=0?'#3FB950':'#EF4444'}/>
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>
      )}
      {data && (
        <div style={{color:'#484F58',fontSize:'0.72rem',marginTop:'0.3rem',textAlign:'center'}}>
          Green = positive preference · Red = negative · 64-dim NCF embedding for User {uid}
        </div>
      )}
    </div>
  )
}

const s: any = {
  card:{background:'#161B2E',border:'1px solid #1F2937',borderRadius:'10px',padding:'1.2rem 1.3rem'},
  sectionTitle:{color:'#8B949E',fontSize:'0.72rem',fontWeight:600,letterSpacing:'0.08em',textTransform:'uppercase'},
  tabBtn:{background:'transparent',border:'1px solid #30363D',color:'#8B949E',borderRadius:'5px',padding:'0.25rem 0.6rem',fontSize:'0.75rem',cursor:'pointer'},
  tabActive:{background:'#1F2D4A',border:'1px solid #388BFD',color:'#58A6FF'},
  input:{background:'#0D1117',border:'1px solid #30363D',color:'#E6EDF3',borderRadius:'6px',padding:'0.35rem 0.6rem',width:'90px',fontSize:'0.85rem'},
  inputLabel:{color:'#8B949E',fontSize:'0.72rem',marginBottom:'0.3rem'},
  btn:{background:'#1F6FEB',color:'white',border:'none',borderRadius:'6px',padding:'0.38rem 0.9rem',fontWeight:600,cursor:'pointer',fontSize:'0.85rem'},
}
