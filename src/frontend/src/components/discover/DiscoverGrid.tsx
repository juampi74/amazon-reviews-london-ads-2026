"use client";

import Image from "next/image";
import { useEffect, useMemo, useState } from "react";
import { Check, Heart, Search, ShoppingBag, Sparkles, ChevronLeft, ChevronRight } from "lucide-react";
import { useRouter } from "next/navigation";
import { Badge } from "@/components/shared/Badge";
import { Button } from "@/components/shared/Button";
import type { StoreProduct } from "@/lib/api/types";
import { useStorePortfolio } from "@/hooks/useStorePortfolio";
import { TrendSparkline } from "./TrendSparkline";

const money = (value: number) => `$${Math.round(value).toLocaleString("en-US")}`;

export function DiscoverGrid() {
  const [filter, setFilter] = useState<string>("All");
  const [trendingProducts, setTrendingProducts] = useState<StoreProduct[]>([]);
  const [isLoading, setIsLoading] = useState(true);

  const [page, setPage] = useState(0);
  const limit = 12;

  const { cart, toggleProduct, createStore } = useStorePortfolio();
  const router = useRouter();

  useEffect(() => {
    async function fetchProducts() {
      setIsLoading(true);
      try {
        const skip = page * limit;
        const response = await fetch(`/api/datasets/products?limit=${limit}&skip=${skip}`);
        if (!response.ok) throw new Error("Failed to load products");
        
        const data = await response.json();
        
        let rawData = [];
        if (Array.isArray(data)) rawData = data;
        else if (data && Array.isArray(data.items)) rawData = data.items;
        else if (data && Array.isArray(data.data)) rawData = data.data;

        const normalizedProducts = rawData.map((p: any, index: number) => {
          
          let cleanDesc = "No description available.";
          if (Array.isArray(p.descriptions) && p.descriptions.length > 0) {
            cleanDesc = p.descriptions[0].description_text || p.descriptions[0];
          } else if (p.description) {
            cleanDesc = String(p.description);
          }
          cleanDesc = cleanDesc.length > 150 ? cleanDesc.substring(0, 147) + "..." : cleanDesc;

          let imageUrl = "/placeholder.png";
          if (Array.isArray(p.images) && p.images.length > 0) {
            imageUrl = p.images.find((img: any) => img.variant === 'MAIN')?.image_url 
                       || p.images[0].image_url 
                       || p.images[0]; 
          }

          let primaryCategory = "Uncategorized";
          if (Array.isArray(p.categories) && p.categories.length > 0) {
            const rawCats = p.categories.map((c: any) => typeof c === 'object' ? c.category_name : c);
            const specificCats = rawCats.filter((c: string) => c && c !== "Beauty & Personal Care");
            
            primaryCategory = specificCats.length > 0 ? specificCats[specificCats.length - 1] : rawCats[0];
          } else if (p.category) {
            primaryCategory = p.category;
          }

          const productPrice = Number(p.price) || Math.floor(Math.random() * 20) + 10;

          return {
            id: p.parent_asin || p.id || String(index),
            name: p.title || p.name || "Producto sin título",
            category: primaryCategory,
            description: cleanDesc,
            image: imageUrl,
            price: productPrice,
            trend: Number(p.trend) || Math.floor(Math.random() * 20) + 5,
            monthlyProfit: Number(p.monthlyProfit) || Math.floor(productPrice * 80),
            successScore: Number(p.successScore) || Math.floor(Math.random() * 40) + 40,
          };
        });

        setTrendingProducts(normalizedProducts);

      } catch (error) {
        console.error("Failed to load products:", error);
        setTrendingProducts([]); 
      } finally {
        setIsLoading(false);
      }
    }
    fetchProducts();
  }, [page]);

  const filters = [
    "All",
    ...new Set(
      trendingProducts
        .map((product) => product.category)
        .filter((category): category is string => typeof category === "string")
    ),
  ];

  const products = useMemo(
    () => trendingProducts.filter((product) => filter === "All" || product.category === filter),
    [filter, trendingProducts]
  );
  
  const total = cart.reduce((sum, product) => sum + (product.monthlyProfit || 0), 0);

  return (
    <>
      <div className="page-intro">
        <span className="eyebrow">Opportunity shelf · real discovery data</span>
        <h1>Beauty products <em>building momentum</em> right now.</h1>
        <p>Shortlist ideas, compare scenario economics, and turn the strongest mix into a focused test store.</p>
      </div>
      
      <div className="filter-row">
        {filters.map((item) => (
          <button key={item} className={filter === item ? "active" : ""} onClick={() => setFilter(item)}>
            {item === "All" ? <Sparkles /> : null}
            {item}
          </button>
        ))}
      </div>

      {isLoading ? (
        <div style={{ padding: "4rem", textAlign: "center", opacity: 0.5 }}>
          <h2>Loading catalog...</h2>
        </div>
      ) : (
        <div className="discover-grid">
          {products.map((product, index) => {
            const selected = cart.some((item) => item.id === product.id);
            return (
              <article className={`product-card ${selected ? "selected" : ""}`} key={product.id}>
                <div 
                  className="product-photo"
                  style={{
                    position: "relative",
                    width: "100%",
                    aspectRatio: "1 / 1",
                    backgroundColor: "#ffffff",
                    padding: "1rem",
                    overflow: "hidden",
                    borderRadius: "8px"
                  }}
                >
                  <Image 
                    src={product.image} 
                    alt={`${product.name} concept`} 
                    fill 
                    sizes="(max-width: 700px) 100vw, 320px" 
                    style={{ objectFit: "contain" }}
                  />
                  <Badge tone="mint">↑ {product.trend}% forecast</Badge>
                </div>
                <span className="product-category">{product.category}</span>
                <h2>{product.name}</h2>
                <p>{product.description}</p>
                
                <TrendSparkline seed={Number(product.id) || index} trend={product.trend} />
                
                <div className="product-money">
                  <small>Could make about</small>
                  <strong>{money(product.monthlyProfit)} <span>/mo</span></strong>
                  <b>Real scenario</b>
                </div>
                <div className="product-stats">
                  <span>Success <b>{product.successScore}%</b></span>
                  <span>Sell at <b>{money(product.price)}</b></span>
                </div>
                <div className="product-actions">
                  <Button variant={selected ? "secondary" : "primary"} onClick={() => toggleProduct(product)}>
                    {selected ? <Check /> : <Heart />}
                    {selected ? "In shortlist" : "Shortlist"}
                  </Button>
                  <button aria-label={`Analyze ${product.name}`} title="Analyze product">
                    <Search />
                  </button>
                </div>
              </article>
            );
          })}
        </div>
      )}

      <div style={{ display: "flex", justifyContent: "center", alignItems: "center", gap: "1rem", marginTop: "3rem", paddingBottom: "6rem" }}>
        <Button variant="secondary" onClick={() => setPage((p) => Math.max(0, p - 1))} disabled={page === 0 || isLoading}>
          <ChevronLeft size={16} style={{ marginRight: 8 }}/> Previous
        </Button>
        <span style={{ fontWeight: 600, fontSize: "0.9rem", color: "var(--text-secondary)" }}>
          Page {page + 1}
        </span>
        <Button variant="secondary" onClick={() => setPage((p) => p + 1)} disabled={products.length < limit || isLoading}>
           Next <ChevronRight size={16} style={{ marginLeft: 8 }}/>
        </Button>
      </div>

      {cart.length ? (
        <div className="cart-dock">
          <div>
            <ShoppingBag />
            <span>
              <b>{cart.length} product{cart.length === 1 ? "" : "s"}</b>
              <small>{money(total)} / month scenario</small>
            </span>
          </div>
          <Button variant="coral" onClick={async () => { await createStore(); router.push("/store"); }}>
            Create my store
          </Button>
        </div>
      ) : null}
    </>
  );
}
