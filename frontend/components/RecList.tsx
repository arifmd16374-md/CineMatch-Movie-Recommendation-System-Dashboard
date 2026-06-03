const GENRE_COLORS: Record<string, { bg: string; color: string }> = {
  Drama:    { bg: 'rgba(124,58,237,0.15)', color: '#A78BFA' },
  Thriller: { bg: 'rgba(210,153,63,0.15)', color: '#D2993F' },
  'Sci-Fi': { bg: 'rgba(88,166,255,0.12)', color: '#58A6FF' },
  Comedy:   { bg: 'rgba(63,185,80,0.15)',  color: '#3FB950' },
  Action:   { bg: 'rgba(63,185,80,0.15)',  color: '#3FB950' },
  Horror:   { bg: 'rgba(239,68,68,0.15)',  color: '#EF4444' },
  Animation:{ bg: 'rgba(249,115,22,0.15)', color: '#F97316' },
}

const AVATAR_COLORS = ['#7C3AED','#D2993F','#388BFD','#3FB950','#EF4444','#F59E0B','#EC4899','#0EA5E9']
const avatarColor = (title: string) => AVATAR_COLORS[title.charCodeAt(0) % AVATAR_COLORS.length]

const PLACEHOLDER = [
  { movie_id: 64,  title: 'The Shawshank Redemption', genre: 'Drama',    predicted_rating: 4.91 },
  { movie_id: 48,  title: 'Pulp Fiction',              genre: 'Thriller', predicted_rating: 4.87 },
  { movie_id: 7,   title: 'Inception',                 genre: 'Sci-Fi',   predicted_rating: 4.84 },
  { movie_id: 127, title: 'The Godfather',             genre: 'Drama',    predicted_rating: 4.82 },
  { movie_id: 258, title: 'Interstellar',              genre: 'Sci-Fi',   predicted_rating: 4.79 },
  { movie_id: 2,   title: 'The Dark Knight',           genre: 'Action',   predicted_rating: 4.76 },
  { movie_id: 273, title: 'Fight Club',                genre: 'Drama',    predicted_rating: 4.73 },
  { movie_id: 37,  title: 'Forrest Gump',              genre: 'Comedy',   predicted_rating: 4.70 },
  { movie_id: 50,  title: 'The Matrix',                genre: 'Sci-Fi',   predicted_rating: 4.68 },
  { movie_id: 174, title: 'Goodfellas',                genre: 'Thriller', predicted_rating: 4.65 },
]

export default function RecList({ recs, model }: { recs: any[]; model: string }) {
  const items = (recs && recs.length > 0) ? recs : PLACEHOLDER
  return (
    <div>
      {items.map((rec, i) => {
        const genre = rec.genre ?? 'Drama'
        const gc = GENRE_COLORS[genre] ?? GENRE_COLORS.Drama
        const color = avatarColor(rec.title)
        const abbr = rec.title.slice(0, 2)
        return (
          <div key={rec.movie_id} style={s.row}>
            <span style={s.rank}>#{i + 1}</span>
            <div style={{ ...s.avatar, background: color }}>{abbr}</div>
            <div style={s.info}>
              <div style={s.title}>{rec.title}</div>
              <span style={{ ...s.genre, background: gc.bg, color: gc.color }}>{genre}</span>
            </div>
            <div style={s.rating}>{rec.predicted_rating.toFixed(2)}</div>
          </div>
        )
      })}
      {recs.length === 0 && (
        <p style={{ color: '#484F58', fontSize: '0.8rem', marginTop: '0.5rem', textAlign: 'center' }}>
          Enter a User ID and click Get recs for live results
        </p>
      )}
    </div>
  )
}

const s: any = {
  row: { display: 'flex', alignItems: 'center', gap: '0.8rem', padding: '0.6rem 0.4rem', borderBottom: '1px solid #1F2937' },
  rank: { color: '#484F58', fontSize: '0.75rem', width: '2rem', flexShrink: 0 },
  avatar: { width: 34, height: 34, borderRadius: '6px', display: 'flex', alignItems: 'center', justifyContent: 'center', color: 'white', fontWeight: 700, fontSize: '0.78rem', flexShrink: 0 },
  info: { flex: 1, minWidth: 0 },
  title: { color: '#E6EDF3', fontSize: '0.86rem', fontWeight: 500, whiteSpace: 'nowrap' as any, overflow: 'hidden', textOverflow: 'ellipsis' },
  genre: { display: 'inline-block', fontSize: '0.68rem', borderRadius: '4px', padding: '0.05rem 0.4rem', marginTop: '0.15rem' },
  rating: { color: '#3ECFCF', fontWeight: 700, fontSize: '1rem', flexShrink: 0 },
}
