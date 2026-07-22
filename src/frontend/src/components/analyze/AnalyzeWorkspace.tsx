"use client";

import { useState, useEffect } from "react";
import { AnalysisForm } from "./AnalysisForm";
import { AnalysisResults } from "./AnalysisResults";
import { Card } from "@/components/shared/Card";
import { EmptyState, ErrorState, LoadingState } from "@/components/shared/States";
import { createDemoAnalysis } from "@/lib/api/demo";
import type { AnalysisRequest, AnalysisResponse } from "@/lib/api/types";

import { useProducts } from "../../hooks/useProducts"; 

const VALID_SUBCATS = [
  "Hair Care", 
  "Skin Care", 
  "Foot, Hand & Nail Care", 
  "Makeup", 
  "Tools & Accessories", 
  "Fragrance", 
  "Shave & Hair Removal", 
  "Personal Care"
];

export function AnalyzeWorkspace() {
  const [analysis, setAnalysis] = useState<AnalysisResponse | null>(null);
  const [input, setInput] = useState<AnalysisRequest | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  
  const [isClient, setIsClient] = useState(false);

  const { products, loading: loadingProducts } = useProducts(50);

  useEffect(() => {
    setIsClient(true);
  }, []);

  const run = async (value: AnalysisRequest) => {
    setBusy(true); setError(null); setInput(value);
    try {
      const response = await fetch("/api/analyses", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(value) });
      if (!response.ok) throw new Error((await response.json().catch(() => null))?.error ?? "The analysis service is unavailable.");
      setAnalysis(await response.json());
    } catch (requestError) {
      if (process.env.NEXT_PUBLIC_ENABLE_DEMO_MODE === "true") setAnalysis(createDemoAnalysis(value));
      else setError(requestError instanceof Error ? requestError.message : "The analysis service is unavailable.");
    } finally { setBusy(false); }
  };

  const runWithRealProduct = () => {
    if (!products || products.length === 0) return;
    
    const randomProduct = products[Math.floor(Math.random() * products.length)];

    let rawCategory = "Skin Care";
    if (Array.isArray(randomProduct.categories) && randomProduct.categories.length > 0) {
      const rawCats = randomProduct.categories.map((c: any) => typeof c === 'object' ? c.category_name : c);
      const specificCats = rawCats.filter((c: string) => c && c !== "Beauty & Personal Care");
      rawCategory = specificCats.length > 0 ? specificCats[specificCats.length - 1] : rawCats[0];
    } else if (randomProduct.category) {
      rawCategory = randomProduct.category;
    }

    const matchedSubcategory = VALID_SUBCATS.includes(rawCategory) ? rawCategory : "Skin Care";

    const rawTitle = String(randomProduct.title || randomProduct.name || "Producto sin título");
    const rawDescription = String(randomProduct.description || "Sin descripción");

    const requestPayload: AnalysisRequest = {
      title: rawTitle.length > 120 ? rawTitle.substring(0, 117) + "..." : rawTitle,
      description: rawDescription.length > 500 ? rawDescription.substring(0, 497) + "..." : rawDescription,
      price: parseFloat(randomProduct.price) || 19.99,
      subcategory: matchedSubcategory,
      risk_preference: "balanced",
    };

    run(requestPayload);
  };

  return <>
    <div className="page-intro">
      <span className="eyebrow">Product decision console</span>
      <h1>Got a beauty product idea? See <em>how strong the evidence is.</em></h1>
      <p>Compare the idea with historical product patterns, inspect the risk, and find a price worth testing.</p>
      
      <div className="mt-6">
        {isClient && (
          <button
            onClick={runWithRealProduct}
            disabled={busy || loadingProducts || products.length === 0}
            className="px-5 py-2.5 bg-violet-600 text-white text-sm font-semibold rounded-lg shadow-sm hover:bg-violet-700 disabled:opacity-50 disabled:cursor-not-allowed transition-all"
          >
            {loadingProducts ? "Connecting to local database..." : "🔮 Autocomplete product..."}
          </button>
        )}
      </div>
    </div>
    
    <div className="analysis-grid">
      <Card accent="violet">
        <AnalysisForm onSubmit={run} busy={busy}/>
      </Card>
      <Card accent="mint" className="results-card">
        {busy ? <LoadingState/> : error ? <ErrorState message={error} onRetry={() => input && run(input)}/> : analysis && input ? <AnalysisResults analysis={analysis} input={input}/> : <EmptyState title="Your launch forecast goes here" message="Fill in the product brief and run the analysis to see model-backed evidence and clearly labeled scenarios."/>}
      </Card>
    </div>
  </>;
}
