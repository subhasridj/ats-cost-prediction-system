import React, {useState} from 'react'

export default function App(){
  const [file, setFile] = useState(null)
  const [result, setResult] = useState(null)

  async function upload(){
    const fd = new FormData()
    fd.append('file', file)
    const res = await fetch('http://localhost:8000/score_tender', {method:'POST', body: fd})
    const j = await res.json()
    setResult(j)
  }

  return (
    <div style={{padding:20}}>
      <h2>ATS Tender Scoring Demo</h2>
      <input type="file" onChange={e=>setFile(e.target.files[0])} />
      <button onClick={upload}>Score Tender</button>
      <pre>{JSON.stringify(result, null, 2)}</pre>
    </div>
  )
}
