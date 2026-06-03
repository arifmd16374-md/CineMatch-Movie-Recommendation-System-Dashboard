import type { Page } from '../pages/index'

const NAV_MAIN = [
  { label: 'Dashboard', icon: '▣', key: 'Dashboard' },
  { label: 'Recommendations', icon: '★', key: 'Recommendations' },
  { label: 'Users', icon: '◉', key: 'Users' },
  { label: 'Items', icon: '▤', key: 'Items' },
]
const NAV_MODEL = [
  { label: 'Evaluation', icon: '◈', key: 'Evaluation' },
  { label: 'Settings', icon: '⚙', key: 'Settings' },
]

export default function Sidebar({ page, setPage, model, setModel }: any) {
  return (
    <aside style={s.sidebar}>
      {/* Brand */}
      <div style={s.brand}>
        <div style={s.dot} />
        <div>
          <div style={s.brandName}>CineMatch</div>
          <div style={s.brandSub}>Movie Recommendation System</div>
        </div>
      </div>

      {/* Nav */}
      <div style={s.section}>MAIN</div>
      {NAV_MAIN.map(n => (
        <button key={n.key} style={{ ...s.navItem, ...(page === n.key ? s.navActive : {}) }}
          onClick={() => setPage(n.key as Page)}>
          <span style={s.navIcon}>{n.icon}</span>
          {n.label}
        </button>
      ))}

      <div style={{ ...s.section, marginTop: '1.2rem' }}>MODEL</div>
      {NAV_MODEL.map(n => (
        <button key={n.key} style={{ ...s.navItem, ...(page === n.key ? s.navActive : {}) }}
          onClick={() => setPage(n.key as Page)}>
          <span style={s.navIcon}>{n.icon}</span>
          {n.label}
        </button>
      ))}

      {/* Model Selector */}
      <div style={{ marginTop: '1.2rem', padding: '0 0.4rem' }}>
        <div style={s.section}>ACTIVE MODEL</div>
        <select value={model} onChange={e => setModel(e.target.value)} style={s.select}>
          <option value="ncf">NCF — Neural CF</option>
          <option value="svd">SVD — Matrix Factor</option>
          <option value="knn">KNN — Neighborhood</option>
        </select>
      </div>

      {/* Status */}
      <div style={s.status}>
        <div style={s.statusDot} />
        <span style={{ color: '#d1dae6', fontSize: '0.82rem' }}>{model.toUpperCase()} · Active</span>
      </div>

      {/* Developer */}
      <div style={s.devCard}>
        <div style={s.devLabel}>DEVELOPED BY</div>
        <div style={s.devName}>Md Arif</div>
      </div>
    </aside>
  )
}

const s: any = {
  sidebar: {
    width: '220px', minWidth: '220px',
    background: 'linear-gradient(180deg,#1e2a3a 0%,#151e2e 100%)',
    borderRight: '1px solid #2d3f54',
    padding: '1rem 0.7rem',
    display: 'flex', flexDirection: 'column',
    height: '100vh', overflowY: 'auto',
  },
  brand: {
    display: 'flex', alignItems: 'center', gap: '0.6rem',
    padding: '0.5rem 0.3rem 1.4rem',
  },
  dot: {
    width: 10, height: 10, borderRadius: '50%',
    background: '#3FB950', boxShadow: '0 0 8px rgba(63,185,80,0.7)',
    flexShrink: 0,
  },
  brandName: {
    fontSize: '1.15rem', fontWeight: 800, letterSpacing: '-0.3px',
    background: 'linear-gradient(135deg,#60a5fa,#a78bfa,#34d399)',
    WebkitBackgroundClip: 'text', WebkitTextFillColor: 'transparent',
    backgroundClip: 'text',
  },
  brandSub: {
    fontSize: '0.65rem', color: '#7a8ba3',
    textTransform: 'uppercase', letterSpacing: '0.06em', marginTop: '2px',
  },
  section: {
    color: '#7a8ba3', fontSize: '0.68rem', fontWeight: 600,
    letterSpacing: '0.1em', textTransform: 'uppercase',
    padding: '0.3rem 0.5rem', marginBottom: '0.3rem',
  },
  navItem: {
    display: 'flex', alignItems: 'center', gap: '0.6rem',
    width: '100%', padding: '0.5rem 0.8rem',
    background: 'transparent', border: 'none',
    borderRadius: '8px', cursor: 'pointer',
    color: '#d1dae6', fontSize: '0.88rem', fontWeight: 500,
    textAlign: 'left', marginBottom: '2px',
    transition: 'background 0.15s',
  },
  navActive: {
    background: 'linear-gradient(90deg,#2d4a6f,#1e3557)',
    color: '#90cdf4', borderLeft: '3px solid #3b82f6',
    paddingLeft: '0.65rem',
  },
  navIcon: { fontSize: '0.9rem', width: '1rem' },
  select: {
    width: '100%', background: '#0D1117',
    border: '1px solid #30363D', color: '#C9D1D9',
    borderRadius: '6px', padding: '0.4rem 0.6rem',
    fontSize: '0.82rem', cursor: 'pointer',
  },
  status: {
    display: 'flex', alignItems: 'center', gap: '0.5rem',
    padding: '0.45rem 0.8rem',
    background: 'rgba(59,130,246,0.1)',
    border: '1px solid rgba(59,130,246,0.2)',
    borderRadius: '8px', marginTop: '1rem',
  },
  statusDot: {
    width: 8, height: 8, borderRadius: '50%',
    background: '#48bb78', boxShadow: '0 0 8px rgba(72,187,120,0.6)',
  },
  devCard: {
    marginTop: '0.8rem', padding: '0.6rem 0.8rem',
    background: 'rgba(16,24,39,0.4)',
    border: '1px solid #2d3f54', borderRadius: '8px',
    textAlign: 'center',
  },
  devLabel: {
    fontSize: '0.65rem', color: '#7a8ba3',
    textTransform: 'uppercase', letterSpacing: '0.08em',
    marginBottom: '0.2rem',
  },
  devName: {
    fontSize: '0.9rem', fontWeight: 700,
    background: 'linear-gradient(135deg,#60a5fa,#3b82f6)',
    WebkitBackgroundClip: 'text', WebkitTextFillColor: 'transparent',
    backgroundClip: 'text',
  },
}
