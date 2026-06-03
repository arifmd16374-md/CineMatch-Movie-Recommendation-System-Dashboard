import { useState, useEffect } from 'react'
import Head from 'next/head'
import Sidebar from '../components/Sidebar'
import Dashboard from '../components/Dashboard'
import Recommendations from '../components/Recommendations'
import Users from '../components/Users'
import Evaluation from '../components/Evaluation'

export type Page = 'Dashboard' | 'Recommendations' | 'Users' | 'Items' | 'Evaluation' | 'Settings'

export default function Home() {
  const [page, setPage] = useState<Page>('Dashboard')
  const [model, setModel] = useState('ncf')
  const [stats, setStats] = useState<any>(null)
  const [metrics, setMetrics] = useState<any>(null)

  useEffect(() => {
    fetch('/api/stats').then(r => r.json()).then(setStats).catch(() => {})
    fetch('/api/metrics').then(r => r.json()).then(setMetrics).catch(() => {})
  }, [])

  return (
    <>
      <Head>
        <title>CineMatch · Movie Recommendations</title>
        <meta name="description" content="CineMatch Movie Recommendation System Dashboard" />
        <link rel="icon" href="/favicon.ico" />
      </Head>
      <div style={{ display: 'flex', height: '100vh', overflow: 'hidden' }}>
        <Sidebar page={page} setPage={setPage} model={model} setModel={setModel} />
        <main style={{ flex: 1, overflow: 'auto', padding: '1.5rem 2rem' }}>
          {page === 'Dashboard'       && <Dashboard model={model} stats={stats} metrics={metrics} />}
          {page === 'Recommendations' && <Recommendations model={model} />}
          {page === 'Users'           && <Users />}
          {page === 'Evaluation'      && <Evaluation metrics={metrics} />}
          {page === 'Items'           && <ItemsPage />}
          {page === 'Settings'        && <SettingsPage />}
        </main>
      </div>
    </>
  )
}

function ItemsPage() {
  return (
    <div>
      <h1 style={styles.pageTitle}>Items</h1>
      <p style={styles.pageSub}>Movie catalog and popularity analysis</p>
      <div style={styles.card}>
        <p style={{ color: '#8B949E' }}>Movie catalog is available after training. Deploy locally to explore all 1,682 movies.</p>
      </div>
    </div>
  )
}

function SettingsPage() {
  return (
    <div>
      <h1 style={styles.pageTitle}>Settings</h1>
      <p style={styles.pageSub}>System configuration</p>
      <div style={styles.card}>
        <div style={{ display: 'flex', flexDirection: 'column', gap: '0.8rem' }}>
          {[
            { label: 'Dataset', value: 'MovieLens 100K', color: '#58A6FF' },
            { label: 'Models', value: 'NCF · SVD · KNN', color: '#3FB950' },
            { label: 'API', value: 'FastAPI + Vercel Serverless', color: '#7C3AED' },
            { label: 'Frontend', value: 'Next.js 14 · TypeScript', color: '#3ECFCF' },
            { label: 'Developer', value: 'Md Arif', color: '#58A6FF' },
          ].map(item => (
            <div key={item.label} style={{ display: 'flex', justifyContent: 'space-between', padding: '0.7rem 1rem', background: '#0D1117', borderRadius: '8px', border: '1px solid #1F2937' }}>
              <span style={{ color: '#8B949E' }}>{item.label}</span>
              <span style={{ color: item.color, fontWeight: 600 }}>{item.value}</span>
            </div>
          ))}
        </div>
      </div>
    </div>
  )
}

const styles: any = {
  pageTitle: { fontSize: '1.7rem', fontWeight: 700, color: '#E6EDF3', marginBottom: '0.25rem' },
  pageSub: { color: '#58A6FF', fontSize: '0.85rem', marginBottom: '1.2rem' },
  card: { background: '#161B2E', border: '1px solid #1F2937', borderRadius: '10px', padding: '1.3rem 1.4rem' },
}
