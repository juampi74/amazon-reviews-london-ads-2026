export type RiskPreference = "cautious" | "balanced" | "bold";
export type SourceType = "model" | "formula" | "external_data" | "simulation";

export interface AnalysisRequest {
  request_id?: string;
  subcategory: string;
  title: string;
  description: string;
  price: number;
  market?: string;
  currency?: string;
  unit_cost?: number;
  fulfilment_cost?: number;
  marketplace_fee_pct?: number;
  advertising_cost_per_unit?: number;
  return_allowance?: number;
  expected_units_monthly?: number;
  risk_preference: RiskPreference;
}

export interface PricePoint {
  price: number;
  score: number;
  profit_per_sale?: number;
}

export interface ComparableProduct {
  parent_asin?: string;
  title: string;
  subcategory?: string;
  price: number | null;
  rating: number | null;
  reviews: number;
  success?: number;
  similarity?: number;
}

export interface ReviewTopic {
  name: string;
  share: number;
}

export interface ReviewTopicsInsight {
  sentiment: { positive: number; neutral: number; negative: number };
  topics: ReviewTopic[];
  positive_keywords: string[];
  negative_keywords: string[];
  rating_distribution?: Record<string, number>;
  verified_purchase_pct?: number;
  sample_size: number;
  dataset_version?: string;
  source_type?: SourceType;
}

export interface AnalysisResponse {
  analysis_id?: number;
  request_id?: string;
  status?: "completed";
  success: {
    score: number;
    probability?: number;
    uncertainty: number;
    confidence: "high" | "medium" | "low" | string;
    source_type?: SourceType;
  };
  risk: {
    index: number;
    components: { downside: number; saturation: number; uncertainty: number };
    source_type?: SourceType;
  };
  saturation: { value: number; source_type?: SourceType };
  recommended_price: number;
  price_range: [number | null, number | null];
  price_curve: PricePoint[];
  comparables: ComparableProduct[];
  topics?: ReviewTopicsInsight | null;
  model_version: string;
  dataset_version: string;
  limitations: string[];
  source: "model" | "demo";
  profit: {
    marketplace_fee: number;
    per_sale: number;
    expected_monthly: number | null;
    is_complete: boolean;
    missing_costs: string[];
    source_type: "formula";
  };
}

export interface StoreProduct {
  id: number | string;
  persistedId?: number;
  key: string;
  category?: string;
  name: string;
  description: string;
  price: number;
  successScore: number;
  monthlyProfit: number;
  startupCost: number;
  image: string;
  trend: number;
  currency?: string;
  sourceType?: SourceType;
}

export interface DemoStore {
  brand: string;
  description: string;
  currency?: string;
  products: StoreProduct[];
}

export interface StoreState {
  store: DemoStore | null;
  shortlist: StoreProduct[];
}
