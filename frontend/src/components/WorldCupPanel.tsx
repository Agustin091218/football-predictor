import { useState, useEffect } from "react";

// Full WC2026 data with all 12 groups
const WC2026 = {
  groups: {
    A: { teams: ["Mexico","South Africa","South Korea","Czechia"], matches: [
      { id:"A1",home:"Mexico",away:"South Africa",date:"2026-06-11",hg:2,ag:0,played:true },
      { id:"A2",home:"South Korea",away:"Czechia",date:"2026-06-11",hg:2,ag:1,played:true },
      { id:"A3",home:"Czechia",away:"South Africa",date:"2026-06-18",hg:null,ag:null,played:false },
      { id:"A4",home:"Mexico",away:"South Korea",date:"2026-06-18",hg:null,ag:null,played:false },
      { id:"A5",home:"Czechia",away:"Mexico",date:"2026-06-24",hg:null,ag:null,played:false },
      { id:"A6",home:"South Africa",away:"South Korea",date:"2026-06-24",hg:null,ag:null,played:false },
    ]},
    B: { teams: ["Canada","Bosnia Herzegovina","Qatar","Switzerland"], matches: [
      { id:"B1",home:"Canada",away:"Bosnia Herzegovina",date:"2026-06-12",hg:1,ag:1,played:true },
      { id:"B2",home:"Qatar",away:"Switzerland",date:"2026-06-12",hg:1,ag:1,played:true },
      { id:"B3",home:"Bosnia Herzegovina",away:"Qatar",date:"2026-06-19",hg:null,ag:null,played:false },
      { id:"B4",home:"Switzerland",away:"Canada",date:"2026-06-19",hg:null,ag:null,played:false },
      { id:"B5",home:"Bosnia Herzegovina",away:"Switzerland",date:"2026-06-24",hg:null,ag:null,played:false },
      { id:"B6",home:"Qatar",away:"Canada",date:"2026-06-25",hg:null,ag:null,played:false },
    ]},
    C: { teams: ["Brazil","Morocco","Haiti","Scotland"], matches: [
      { id:"C1",home:"Brazil",away:"Morocco",date:"2026-06-13",hg:1,ag:1,played:true },
      { id:"C2",home:"Haiti",away:"Scotland",date:"2026-06-13",hg:null,ag:null,played:false },
      { id:"C3",home:"Morocco",away:"Haiti",date:"2026-06-19",hg:null,ag:null,played:false },
      { id:"C4",home:"Scotland",away:"Brazil",date:"2026-06-20",hg:null,ag:null,played:false },
      { id:"C5",home:"Morocco",away:"Scotland",date:"2026-06-25",hg:null,ag:null,played:false },
      { id:"C6",home:"Brazil",away:"Haiti",date:"2026-06-25",hg:null,ag:null,played:false },
    ]},
    D: { teams: ["United States","Paraguay","Australia","Iraq"], matches: [
      { id:"D1",home:"United States",away:"Paraguay",date:"2026-06-12",hg:4,ag:1,played:true },
      { id:"D2",home:"Australia",away:"Iraq",date:"2026-06-13",hg:2,ag:0,played:true },
      { id:"D3",home:"Paraguay",away:"Australia",date:"2026-06-19",hg:null,ag:null,played:false },
      { id:"D4",home:"Iraq",away:"United States",date:"2026-06-20",hg:null,ag:null,played:false },
      { id:"D5",home:"Paraguay",away:"Iraq",date:"2026-06-25",hg:null,ag:null,played:false },
      { id:"D6",home:"Australia",away:"United States",date:"2026-06-25",hg:null,ag:null,played:false },
    ]},
    E: { teams: ["Germany","Cura\u00e7ao","Ivory Coast","Ecuador"], matches: [
      { id:"E1",home:"Germany",away:"Cura\u00e7ao",date:"2026-06-14",hg:null,ag:null,played:false },
      { id:"E2",home:"Ivory Coast",away:"Ecuador",date:"2026-06-15",hg:null,ag:null,played:false },
      { id:"E3",home:"Germany",away:"Ivory Coast",date:"2026-06-21",hg:null,ag:null,played:false },
      { id:"E4",home:"Ecuador",away:"Cura\u00e7ao",date:"2026-06-21",hg:null,ag:null,played:false },
      { id:"E5",home:"Germany",away:"Ecuador",date:"2026-06-26",hg:null,ag:null,played:false },
      { id:"E6",home:"Cura\u00e7ao",away:"Ivory Coast",date:"2026-06-26",hg:null,ag:null,played:false },
    ]},
    F: { teams: ["Netherlands","Japan","Sweden","Tunisia"], matches: [
      { id:"F1",home:"Netherlands",away:"Japan",date:"2026-06-14",hg:null,ag:null,played:false },
      { id:"F2",home:"Sweden",away:"Tunisia",date:"2026-06-15",hg:null,ag:null,played:false },
      { id:"F3",home:"Japan",away:"Sweden",date:"2026-06-21",hg:null,ag:null,played:false },
      { id:"F4",home:"Tunisia",away:"Netherlands",date:"2026-06-22",hg:null,ag:null,played:false },
      { id:"F5",home:"Japan",away:"Tunisia",date:"2026-06-26",hg:null,ag:null,played:false },
      { id:"F6",home:"Netherlands",away:"Sweden",date:"2026-06-26",hg:null,ag:null,played:false },
    ]},
    G: { teams: ["Belgium","Egypt","Iran","New Zealand"], matches: [
      { id:"G1",home:"Belgium",away:"Egypt",date:"2026-06-16",hg:null,ag:null,played:false },
      { id:"G2",home:"Iran",away:"New Zealand",date:"2026-06-16",hg:null,ag:null,played:false },
      { id:"G3",home:"Egypt",away:"Iran",date:"2026-06-22",hg:null,ag:null,played:false },
      { id:"G4",home:"New Zealand",away:"Belgium",date:"2026-06-22",hg:null,ag:null,played:false },
      { id:"G5",home:"Egypt",away:"New Zealand",date:"2026-06-27",hg:null,ag:null,played:false },
      { id:"G6",home:"Belgium",away:"Iran",date:"2026-06-27",hg:null,ag:null,played:false },
    ]},
    H: { teams: ["Spain","Cape Verde","Saudi Arabia","Uruguay"], matches: [
      { id:"H1",home:"Spain",away:"Cape Verde",date:"2026-06-16",hg:null,ag:null,played:false },
      { id:"H2",home:"Saudi Arabia",away:"Uruguay",date:"2026-06-16",hg:null,ag:null,played:false },
      { id:"H3",home:"Cape Verde",away:"Saudi Arabia",date:"2026-06-22",hg:null,ag:null,played:false },
      { id:"H4",home:"Uruguay",away:"Spain",date:"2026-06-22",hg:null,ag:null,played:false },
      { id:"H5",home:"Cape Verde",away:"Uruguay",date:"2026-06-27",hg:null,ag:null,played:false },
      { id:"H6",home:"Spain",away:"Saudi Arabia",date:"2026-06-27",hg:null,ag:null,played:false },
    ]},
    I: { teams: ["France","Senegal","DR Congo","Norway"], matches: [
      { id:"I1",home:"France",away:"Senegal",date:"2026-06-17",hg:null,ag:null,played:false },
      { id:"I2",home:"DR Congo",away:"Norway",date:"2026-06-17",hg:null,ag:null,played:false },
      { id:"I3",home:"Senegal",away:"DR Congo",date:"2026-06-23",hg:null,ag:null,played:false },
      { id:"I4",home:"Norway",away:"France",date:"2026-06-23",hg:null,ag:null,played:false },
      { id:"I5",home:"Senegal",away:"Norway",date:"2026-06-27",hg:null,ag:null,played:false },
      { id:"I6",home:"France",away:"DR Congo",date:"2026-06-27",hg:null,ag:null,played:false },
    ]},
    J: { teams: ["Argentina","Algeria","Austria","Jordan"], matches: [
      { id:"J1",home:"Argentina",away:"Algeria",date:"2026-06-17",hg:null,ag:null,played:false },
      { id:"J2",home:"Austria",away:"Jordan",date:"2026-06-17",hg:null,ag:null,played:false },
      { id:"J3",home:"Algeria",away:"Austria",date:"2026-06-23",hg:null,ag:null,played:false },
      { id:"J4",home:"Jordan",away:"Argentina",date:"2026-06-23",hg:null,ag:null,played:false },
      { id:"J5",home:"Algeria",away:"Jordan",date:"2026-06-27",hg:null,ag:null,played:false },
      { id:"J6",home:"Argentina",away:"Austria",date:"2026-06-27",hg:null,ag:null,played:false },
    ]},
    K: { teams: ["Portugal","DR Congo","Uzbekistan","Colombia"], matches: [
      { id:"K1",home:"Portugal",away:"DR Congo",date:"2026-06-17",hg:null,ag:null,played:false },
      { id:"K2",home:"Uzbekistan",away:"Colombia",date:"2026-06-17",hg:null,ag:null,played:false },
      { id:"K3",home:"DR Congo",away:"Uzbekistan",date:"2026-06-23",hg:null,ag:null,played:false },
      { id:"K4",home:"Colombia",away:"Portugal",date:"2026-06-23",hg:null,ag:null,played:false },
      { id:"K5",home:"DR Congo",away:"Colombia",date:"2026-06-28",hg:null,ag:null,played:false },
      { id:"K6",home:"Portugal",away:"Uzbekistan",date:"2026-06-28",hg:null,ag:null,played:false },
    ]},
    L: { teams: ["England","Croatia","Ghana","Panama"], matches: [
      { id:"L1",home:"England",away:"Croatia",date:"2026-06-17",hg:null,ag:null,played:false },
      { id:"L2",home:"Ghana",away:"Panama",date:"2026-06-17",hg:null,ag:null,played:false },
      { id:"L3",home:"Croatia",away:"Ghana",date:"2026-06-23",hg:null,ag:null,played:false },
      { id:"L4",home:"Panama",away:"England",date:"2026-06-23",hg:null,ag:null,played:false },
      { id:"L5",home:"Croatia",away:"Panama",date:"2026-06-28",hg:null,ag:null,played:false },
      { id:"L6",home:"England",away:"Ghana",date:"2026-06-28",hg:null,ag:null,played:false },
    ]},
  },
};

const RESULTS_KEY = "wc2026_results";
const CACHE_KEY = "wc2026_preds";

function loadResults(): Record<string, { homeGoals: number; awayGoals: number }> {
  try { return JSON.parse(localStorage.getItem(RESULTS_KEY) || "{}"); } catch { return {}; }
}

function getCachedPrediction(home: string, away: string) {
  try {
    const cache = JSON.parse(localStorage.getItem(CACHE_KEY) || "{}");
    for (const v of Object.values(cache) as { home: string; away: string; prediction?: { prob_home_win?: number; prob_draw?: number; prob_away_win?: number; confidence?: number } }[]) {
      if (v.home === home && v.away === away) return v;
    }
  } catch {}
  return null;
}

function calcStandings(group: typeof WC2026.groups.A) {
  const teams: Record<string, { team: string; pj: number; w: number; d: number; l: number; gf: number; gc: number; pts: number }> = {};
  for (const t of group.teams) teams[t] = { team: t, pj: 0, w: 0, d: 0, l: 0, gf: 0, gc: 0, pts: 0 };
  for (const m of group.matches) {
    if (!m.played || m.hg == null) continue;
    const h = teams[m.home], a = teams[m.away];
    h.pj++; a.pj++; h.gf += m.hg; h.gc += m.ag; a.gf += m.ag; a.gc += m.hg;
    if (m.hg > m.ag) { h.w++; h.pts += 3; a.l++; }
    else if (m.hg === m.ag) { h.d++; h.pts++; a.d++; a.pts++; }
    else { a.w++; a.pts += 3; h.l++; }
  }
  return Object.values(teams).sort((a, b) => b.pts - a.pts || (b.gf - b.gc) - (a.gf - a.gc) || b.gf - a.gf);
}

export function WorldCupPanel({ onPredict, onAddResult }: { onPredict: (home: string, away: string) => void; onAddResult: (matchId: string, home: string, away: string) => void }) {
  const [activeGroup, setActiveGroup] = useState("A");
  const groupKeys = Object.keys(WC2026.groups).sort();

  useEffect(() => {
    const saved = loadResults();
    for (const [mid, data] of Object.entries(saved)) {
      for (const g of Object.values(WC2026.groups)) {
        const m = g.matches.find((m) => m.id === mid);
        if (m) { m.hg = data.homeGoals; m.ag = data.awayGoals; m.played = true; }
      }
    }
  }, []);

  return (
    <div className="flex gap-4 p-4">
      <div className="w-40 flex-shrink-0 space-y-1 max-h-[calc(100vh-100px)] overflow-y-auto sticky top-20">
        {groupKeys.map((gid) => {
          const g = WC2026.groups[gid as keyof typeof WC2026.groups];
          const played = g.matches.filter((m) => m.played).length;
          return (
            <button key={gid} onClick={() => setActiveGroup(gid)}
              className={`w-full text-left px-3 py-1.5 rounded text-xs flex justify-between ${activeGroup === gid ? "bg-[#1C2333] text-gray-200 border border-[#252D3D]" : "text-gray-500 hover:text-gray-300"}`}>
              Grupo {gid}<span className="text-[10px] text-gray-600">{played}/{g.matches.length}</span>
            </button>
          );
        })}
      </div>
      <div className="flex-1">
        {groupKeys.map((gid) => {
          if (gid !== activeGroup) return null;
          const g = WC2026.groups[gid as keyof typeof WC2026.groups];
          const st = calcStandings(g);
          return (
            <div key={gid} className="bg-[#161B26] border border-[#252D3D] rounded-xl p-4">
              <h3 className="text-sm font-semibold mb-3">Grupo {gid} · {g.teams.join(", ")}</h3>
              <div className="overflow-x-auto">
                <table className="w-full text-xs">
                  <thead><tr className="text-gray-500 border-b border-[#1a2030]"><th className="text-left py-1">#</th><th className="text-left">Equipo</th><th>PJ</th><th>G</th><th>E</th><th>P</th><th>GF</th><th>GC</th><th>DG</th><th>Pts</th></tr></thead>
                  <tbody>
                    {st.map((t, i) => (
                      <tr key={t.team} className={`border-b border-[#1a2030] ${i < 2 ? "bg-emerald-950/20" : ""}`}>
                        <td className="py-1">{i + 1}</td><td>{t.team}</td><td className="text-center">{t.pj}</td><td className="text-center">{t.w}</td><td className="text-center">{t.d}</td><td className="text-center">{t.l}</td><td className="text-center">{t.gf}</td><td className="text-center">{t.gc}</td><td className="text-center font-mono">{t.gf - t.gc >= 0 ? "+" : ""}{t.gf - t.gc}</td><td className="text-center font-bold">{t.pts}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
              <div className="mt-4 space-y-1">
                {g.matches.map((m) => {
                  const cached = getCachedPrediction(m.home, m.away);
                  return (
                    <div key={m.id} className="flex items-center justify-between text-xs py-1.5 border-b border-[#1a2030] gap-2">
                      <span className="w-44">{m.home} <span className="text-gray-600">vs</span> {m.away}</span>
                      <span className="text-gray-500 text-[10px] w-16 text-center">{new Date(m.date + "T00:00:00").toLocaleDateString("es-AR", { day: "2-digit", month: "short" })}</span>
                      {m.played && m.hg != null ? (
                        <span className="font-mono font-bold w-12 text-center">{m.hg} — {m.ag}</span>
                      ) : cached ? (
                        <span className="text-emerald-400 text-[10px] w-36 text-center">
                          🤖 {(cached.prediction?.prob_home_win ?? 0) > (cached.prediction?.prob_away_win ?? 0) ? m.home : m.away} {(Math.max(cached.prediction?.prob_home_win ?? 0, cached.prediction?.prob_draw ?? 0, cached.prediction?.prob_away_win ?? 0) * 100).toFixed(0)}%
                        </span>
                      ) : null}
                      <button onClick={() => onPredict(m.home, m.away)} className="bg-emerald-500 hover:bg-emerald-400 text-black px-2 py-0.5 rounded text-[10px] font-semibold whitespace-nowrap">
                        {cached ? "Ver" : "Predecir →"}
                      </button>
                      {!m.played && (
                        <button onClick={() => onAddResult(m.id, m.home, m.away)} className="text-gray-500 hover:text-gray-300 text-xs px-1" title="Ingresar resultado">📝</button>
                      )}
                    </div>
                  );
                })}
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
