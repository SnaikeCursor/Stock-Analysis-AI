import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import './index.css'
import App from './App.tsx'

// GitHub Pages SPA workaround: 404.html redirects here with the original
// path encoded in ?p=, so we restore it before React mounts.
;(() => {
  const params = new URLSearchParams(location.search)
  const redirect = params.get('p')
  if (redirect) {
    params.delete('p')
    const qs = params.toString()
    const target = redirect + (qs ? `?${qs}` : '') + location.hash
    history.replaceState(null, '', target)
  }
})()

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <App />
  </StrictMode>,
)
