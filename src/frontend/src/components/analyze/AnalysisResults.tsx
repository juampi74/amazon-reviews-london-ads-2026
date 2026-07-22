"use client";

import { useMemo, useState } from "react";
import { ChevronDown, CircleDollarSign, Info, Lightbulb, MessageCircleMore, PackageSearch, Rocket, RotateCcw, Search, ShieldAlert, Sparkles, Tags, Target, ThermometerSun, UsersRound } from "lucide-react";
import type { AnalysisRequest, AnalysisResponse } from "@/lib/api/types";
import { Badge } from "@/components/shared/Badge";
import { AudienceChart, ForecastChart, PriceSuccessSimulator } from "./AnalysisCharts";

const formatMoney = (value: number, currency: string) => new Intl.NumberFormat("en-US", { style: "currency", currency, maximumFractionDigits: 0 }).format(value);

const categoryTopics: Record<string, string[]> = {
  "Hair Care": ["Results and growth", "Scent", "Texture", "Ease of use"],
  "Skin Care": ["Hydration", "Skin feel", "Visible results", "Ingredients"],
  "Foot, Hand & Nail Care": ["Strength and growth", "Absorption", "Application", "Scent"],
  Makeup: ["Colour payoff", "Wear time", "Texture", "Packaging"],
  "Tools & Accessories": ["Ease of use", "Build quality", "Results", "Cleaning"],
  Fragrance: ["Scent profile", "Longevity", "Projection", "Packaging"],
  "Shave & Hair Removal": ["Skin comfort", "Results", "Ease of use", "Durability"],
  "Personal Care": ["Effectiveness", "Scent", "Gentleness", "Value"],
};

export function AnalysisResults({ analysis, input }: { analysis: AnalysisResponse; input: AnalysisRequest }) {
  const [expanded, setExpanded] = useState(false);
  const money = (value: number) => formatMoney(value, input.currency ?? "USD");
  const score = analysis.success.score;
  const estimatedMargin = analysis.profit.per_sale;
  const estimatedUnits = input.expected_units_monthly ?? Math.round(35 + score * 1.45);
  const monthly = analysis.profit.expected_monthly ?? Math.round(estimatedMargin * estimatedUnits);
  const verdict = score >= 75
    ? { label: "Strong test candidate", icon: Rocket, tone: "high" }
    : score >= 50
      ? { label: "Promising, with conditions", icon: Lightbulb, tone: "mid" }
      : { label: "Rework before investing", icon: RotateCcw, tone: "low" };
  const VerdictIcon = verdict.icon;
  const sweetRange = analysis.price_range.filter((value): value is number => value !== null);
  const sources = useMemo(() => analysis.source === "model"
    ? [{ label: "Success, risk and market", tone: "mint" as const }, { label: "Profit and demand", tone: "sun" as const }]
    : [{ label: "All panels use demo data", tone: "sun" as const }], [analysis.source]);
  const saturation = Math.max(0, Math.min(100, Math.round(analysis.saturation.value)));
  const saturationLabel = saturation >= 66 ? "High" : saturation >= 40 ? "Moderate" : "Low";
  const saturationNote = saturation >= 66
    ? "Crowded space. You need a clear hook and strong reviews to stand out."
    : saturation >= 40
      ? "Busy but not full. A distinctive product can still get noticed."
      : "Open space. There is room to become a go-to option.";
  const topics = (categoryTopics[input.subcategory] ?? categoryTopics["Personal Care"]).map((name, index) => ({ name, share: [34, 26, 18, 12][index] }));
  const curveMin = Math.floor(analysis.price_curve[0]?.price ?? Math.max(1, input.price * 0.5));
  const curveMax = Math.ceil(analysis.price_curve.at(-1)?.price ?? input.price * 1.5);
  const rangeSpan = Math.max(1, curveMax - curveMin);
  const rangeStart = Math.max(0, Math.min(100, (((sweetRange[0] ?? curveMin) - curveMin) / rangeSpan) * 100));
  const rangeEnd = Math.max(rangeStart, Math.min(100, (((sweetRange[1] ?? curveMax) - curveMin) / rangeSpan) * 100));
  const priceMarker = Math.max(0, Math.min(100, ((input.price - curveMin) / rangeSpan) * 100));

  return <div className="results-content">
    <div className="result-topline"><div><span className="eyebrow">Your forecast</span><h2>{input.title}</h2></div><div className="source-stack">{sources.map((item) => <Badge key={item.label} tone={item.tone}>{item.label}</Badge>)}</div></div>
    <section className="money-hero"><p><CircleDollarSign/> {analysis.profit.is_complete ? "Cost-complete scenario profit" : "Estimated scenario profit"}</p><strong>{money(monthly)} <small>/month</small></strong><span>{analysis.profit.is_complete ? "Calculated from the costs and unit scenario you supplied." : "Some costs are missing; this remains a clearly labeled estimate."}</span><div><b>{money(monthly * 12)} / year</b><b>{estimatedUnits} units / month</b></div></section>
    <section className={`verdict ${verdict.tone}`}><VerdictIcon/><span><strong>{verdict.label}</strong><small>{score}% calibrated historical success proxy · {analysis.success.confidence} confidence</small></span></section>
    <div className="metric-grid">
      <article><Target/><strong>{score}%</strong><span>Success Score</span><small>Calibrated proxy</small></article>
      <article><ShieldAlert/><strong>{analysis.risk.index}</strong><span>Decision Risk</span><small>Index, not probability</small></article>
      <article><CircleDollarSign/><strong>{money(estimatedMargin)}</strong><span>Profit per sale</span><small>Formula estimate</small></article>
    </div>
    <section className="recommendation"><div className="section-title"><Sparkles/> Launch recommendation</div><p>Test at <b>{money(analysis.recommended_price)}</b> with a small batch. Your strongest evidence is the model-backed score; the profit panel remains a planning assumption until costs and expected units are supplied.</p><div className="recommendation-grid"><span><small>Suggested range</small><b>{sweetRange.length === 2 ? `${money(sweetRange[0])}–${money(sweetRange[1])}` : "Not available"}</b></span><span><small>Market saturation</small><b>{saturation} / 100</b></span><span><small>Model confidence</small><b>{analysis.success.confidence}</b></span></div></section>

    <PriceSuccessSimulator key={`${input.subcategory}-${input.price}-${analysis.model_version}`} data={analysis.price_curve} currentPrice={input.price} saturation={saturation}/>

    <div className="reason-list"><div className="section-title"><Info/> Why Launchly is saying this</div><article className="reason-positive">✓ Your price sits {input.price >= (sweetRange[0] ?? 0) && input.price <= (sweetRange[1] ?? Infinity) ? "inside" : "outside"} the category&apos;s supported range.</article><article className="reason-neutral">○ Similarity and saturation are comparative market signals, not market-share estimates.</article><article className="reason-negative">! The model has moderate predictive signal; use the score to plan a test, not to guarantee sales.</article></div>
    <button className="breakdown-toggle" onClick={() => setExpanded((value) => !value)} aria-expanded={expanded}><span>{expanded ? "Hide the full breakdown" : "See the full breakdown"}</span><ChevronDown className={expanded ? "rotated" : ""}/></button>

    {expanded ? <div className="breakdown">
      <section><div className="section-title"><Sparkles/> Demand forecast <Badge tone="sun">Simulated</Badge></div><ForecastChart/></section>

      <section className="detail-section sweet-spot-section">
        <div className="section-title"><Tags/> Sweet spot price range</div>
        <div className="detail-box">
          <div className="sweet-price">{sweetRange.length === 2 ? money(sweetRange[0]) : "N/A"} <span>to {sweetRange.length === 2 ? money(sweetRange[1]) : "N/A"}</span></div>
          <div className="price-range-track"><span className="price-range-fill" style={{ left: `${rangeStart}%`, right: `${100 - rangeEnd}%` }}/><span className="price-range-marker" style={{ left: `${priceMarker}%` }} data-label="You"/></div>
          <div className="range-end-labels"><span>{money(curveMin)}</span><span>{money(curveMax)}</span></div>
        </div>
      </section>

      <section className="detail-section saturation-section">
        <div className="section-title"><ThermometerSun/> How crowded is this market</div>
        <div className="detail-box"><div className="saturation-head"><strong>{saturationLabel}</strong><b>{saturation}%</b></div><div className="saturation-track"><span style={{ width: `${saturation}%` }}/></div><p><b>What this means:</b> {saturationNote}</p></div>
      </section>

      <section><div className="section-title"><UsersRound/> Who buys this <Badge tone="sun">Simulated</Badge></div><AudienceChart/></section>

      <section className="detail-section topics-section">
        <div className="section-title"><MessageCircleMore/> What customers talk about most <Badge tone="sun">Category scenario</Badge></div>
        <div className="detail-box topic-list">{topics.map((topic) => <div className="topic-row" key={topic.name}><div><b>{topic.name}</b><span>{topic.share}%</span></div><div><span style={{ width: `${topic.share}%` }}/></div></div>)}</div>
      </section>

      <section className="detail-section comparable-section">
        <div className="section-title"><Search/> Similar products we compared against <Badge tone={analysis.source === "model" ? "mint" : "sun"}>{analysis.source === "model" ? "Model" : "Demo"}</Badge></div>
        <div className="comparable-product-grid">{analysis.comparables.slice(0, 5).map((product, index) => <article key={`${product.title}-${index}`}><div className="comparable-icon"><PackageSearch/></div><h3>{product.title}</h3><div><b>{product.price ? money(product.price) : "No price"}</b><span>{product.rating ? `★ ${product.rating.toFixed(1)}` : "No rating"}</span></div><small>{product.similarity ? `${Math.round(product.similarity * 100)}% similar` : `${product.reviews.toLocaleString()} reviews`}</small></article>)}</div>
      </section>
    </div> : null}
  </div>;
}
