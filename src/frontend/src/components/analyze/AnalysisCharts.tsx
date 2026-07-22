"use client";

import { useMemo, useState } from "react";
import { SlidersHorizontal } from "lucide-react";
import { Area, AreaChart, Bar, BarChart, CartesianGrid, Cell, Legend, Line, LineChart, ReferenceDot, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import type { PricePoint } from "@/lib/api/types";

const tooltipStyle = { borderRadius: 14, border: "1px solid #efe5f5", fontFamily: "var(--font-body)", fontWeight: 800 };
const money = (value: number) => new Intl.NumberFormat("en-US", { style: "currency", currency: "USD", maximumFractionDigits: 0 }).format(value);

export function PriceSuccessSimulator({ data, currentPrice, saturation }: { data: PricePoint[]; currentPrice: number; saturation: number }) {
  const ordered = useMemo(() => [...data].sort((a, b) => a.price - b.price), [data]);
  const min = Math.floor(ordered[0]?.price ?? 1);
  const max = Math.ceil(ordered.at(-1)?.price ?? Math.max(100, currentPrice));
  const initial = Math.max(min, Math.min(max, Math.round(currentPrice)));
  const [selectedPrice, setSelectedPrice] = useState(initial);
  const selectedPoint = useMemo(() => ordered.reduce((closest, point) => Math.abs(point.price - selectedPrice) < Math.abs(closest.price - selectedPrice) ? point : closest, ordered[0] ?? { price: selectedPrice, score: 0 }), [ordered, selectedPrice]);
  const bestPoint = useMemo(() => ordered.reduce((best, point) => point.score > best.score ? point : best, ordered[0] ?? { price: selectedPrice, score: 0 }), [ordered, selectedPrice]);
  const estimatedUnits = Math.round(35 + selectedPoint.score * 1.45);
  const monthlyProfit = Math.round(selectedPrice * 0.51 * estimatedUnits);
  const decisionRisk = Math.max(0, Math.min(100, Math.round(0.58 * (100 - selectedPoint.score) + 0.42 * saturation)));
  const nearBest = Math.abs(selectedPrice - bestPoint.price) <= 2;

  const updatePrice = (value: number) => setSelectedPrice(Math.max(min, Math.min(max, Number.isFinite(value) ? value : min)));

  return <section className="profit-simulator">
    <div className="simulator-heading"><SlidersHorizontal/><div><h3>Price vs. success <span>(find your money-making price)</span></h3><p>Slide the price. The dot rides the curve so you see success and profit at every price.</p></div></div>
    <div className="price-curve-chart" aria-label="Success score by selling price">
      <ResponsiveContainer width="100%" height={190}>
        <AreaChart data={ordered} margin={{ top: 25, right: 16, left: -18, bottom: 0 }}>
          <defs><linearGradient id="interactiveScoreFill" x1="0" x2="0" y1="0" y2="1"><stop offset="0" stopColor="#12c296" stopOpacity={0.28}/><stop offset="1" stopColor="#12c296" stopOpacity={0}/></linearGradient></defs>
          <CartesianGrid stroke="#f0e4f5" strokeDasharray="4 4" vertical={false}/>
          <XAxis type="number" dataKey="price" domain={[min, max]} tickFormatter={(value) => `$${Math.round(value)}`} tick={{ fill: "#8c839d", fontSize: 10, fontWeight: 800 }} tickLine={false} axisLine={false}/>
          <YAxis domain={[0, 100]} tickFormatter={(value) => `${value}%`} tick={{ fill: "#8c839d", fontSize: 10, fontWeight: 800 }} tickLine={false} axisLine={false}/>
          <Tooltip contentStyle={tooltipStyle} labelFormatter={(value) => money(Number(value))} formatter={(value) => [`${value}%`, "Success score"]}/>
          <ReferenceLine x={bestPoint.price} stroke="#12c296" strokeDasharray="4 4" label={{ value: "best", position: "insideTop", fill: "#087d60", fontSize: 10, fontWeight: 900 }}/>
          <Area type="monotone" dataKey="score" stroke="#12c296" strokeWidth={3.5} fill="url(#interactiveScoreFill)" activeDot={{ r: 5 }}/>
          <ReferenceDot x={selectedPoint.price} y={selectedPoint.score} r={7} fill="#ff6b5b" stroke="#fff" strokeWidth={3} label={{ value: `${selectedPoint.score}%`, position: "top", fill: "#df493a", fontSize: 11, fontWeight: 900 }}/>
        </AreaChart>
      </ResponsiveContainer>
    </div>
    <div className="price-control-row">
      <input className="price-success-slider" type="range" min={min} max={max} step="1" value={selectedPrice} onChange={(event) => updatePrice(Number(event.target.value))} aria-label="Test a selling price"/>
      <label className="simulator-price-input"><span>$</span><input type="number" min={min} max={max} step="1" value={selectedPrice} onChange={(event) => updatePrice(Number(event.target.value))} aria-label="Selling price"/></label>
    </div>
    <div className="simulator-end-labels"><span>{money(min)}</span><span>{money(max)}</span></div>
    <div className="simulator-readout">At <b>{money(selectedPrice)}</b>: <b>{selectedPoint.score}%</b> success, <b>{decisionRisk}%</b> risk, about <b>{money(monthlyProfit)}/month</b>{nearBest ? <strong> Best price zone</strong> : null}</div>
    <small className="simulator-source">Success follows the model price curve. Risk and profit are transparent planning formulas.</small>
  </section>;
}

interface ForecastChartProps {
  growth?: number;
  categoryName?: string;
}

export function ForecastChart({ growth, categoryName }: ForecastChartProps) {
  const chartData = useMemo(() => {
    const baseGrowth = growth ?? 25;
    
    let hash = 0;
    if (categoryName) {
      for (let i = 0; i < categoryName.length; i++) {
        hash = (hash * 31 + categoryName.charCodeAt(i)) % 1000;
      }
    }

    const months = ["Aug", "Sep", "Oct", "Nov", "Dec", "Jan", "Feb", "Mar", "Apr"];

    return months.map((month, index) => {
      const isHistory = ["Aug", "Sep", "Oct", "Nov", "Dec", "Jan"].includes(month);
      const isForecast = ["Jan", "Feb", "Mar", "Apr"].includes(month);

      const categoryWave = Math.sin((index + 1) * (1 + (hash % 3))) * 12 + (((hash + index) % 4) - 1.5) * 3;
      const historyVal = Math.round(Math.min(90, Math.max(15, (45 + categoryWave))));

      let forecastVal = null;
      if (isForecast) {
        if (month === "Jan") {
          forecastVal = historyVal;
        } else {
          const step = index - 5; 
          const growthBoost = (baseGrowth / 15) * (step * 8);
          forecastVal = Math.round(Math.min(98, Math.max(historyVal, historyVal + growthBoost + (Math.sin(hash + index) * 4))));
        }
      }

      return {
        month,
        history: isHistory ? historyVal : null,
        forecast: forecastVal,
      };
    });
  }, [growth, categoryName]);

  return (
    <div className="chart-wrap" aria-label="Simulated demand history and three month forecast">
      <ResponsiveContainer width="100%" height={220}>
        <LineChart data={chartData} margin={{ top: 8, right: 12, left: -28, bottom: 0 }}>
          <CartesianGrid stroke="#f0e4f5" strokeDasharray="4 4" vertical={false}/>
          <XAxis dataKey="month" tick={{ fill: "#8c839d", fontSize: 11 }}/>
          <YAxis domain={[0, 100]} tick={{ fill: "#8c839d", fontSize: 11 }}/>
          <Tooltip contentStyle={tooltipStyle}/>
          <Legend/>
          <Line type="monotone" dataKey="history" name="Interest" stroke="#8c6bff" strokeWidth={3} dot={false} connectNulls={false}/>
          <Line type="monotone" dataKey="forecast" name="Forecast" stroke="#0f9e74" strokeWidth={3} strokeDasharray="6 5" dot={false} connectNulls={true}/>
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}

const audience = [{ age: "18–24", share: 17 }, { age: "25–34", share: 32 }, { age: "35–44", share: 27 }, { age: "45–54", share: 15 }, { age: "55+", share: 9 }];
export function AudienceChart() {
  return <div className="chart-wrap" aria-label="Simulated audience age distribution"><ResponsiveContainer width="100%" height={210}><BarChart data={audience} margin={{ top: 8, right: 5, left: -32, bottom: 0 }}><CartesianGrid stroke="#f0e4f5" strokeDasharray="4 4" vertical={false}/><XAxis dataKey="age" tick={{ fill: "#8c839d", fontSize: 10 }}/><YAxis tick={{ fill: "#8c839d", fontSize: 10 }}/><Tooltip contentStyle={tooltipStyle} formatter={(value) => [`${value}%`, "Share"]}/><Bar dataKey="share" radius={[8, 8, 2, 2]}>{audience.map((_, index) => <Cell key={index} fill={index === 1 ? "#6c46f0" : "#a78bfa"}/>)}</Bar></BarChart></ResponsiveContainer></div>;
}
