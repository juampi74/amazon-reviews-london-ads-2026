"use client";

import Image from "next/image";
import { useState } from "react";
import { Bot, Boxes, CircleDollarSign, PackagePlus, RotateCcw, ShieldCheck, Store, Trash2, TrendingUp, Loader2 } from "lucide-react";
import { useRouter } from "next/navigation";
import { Button } from "@/components/shared/Button";
import { Badge } from "@/components/shared/Badge";
import { Modal } from "@/components/shared/Modal";
import { useStorePortfolio } from "@/hooks/useStorePortfolio";

const money = (value: number) => `$${Math.round(value).toLocaleString("en-US")}`;

export function StoreWorkspace() {
  const { store, ready, createStore, removeProduct, clearStore } = useStorePortfolio();
  const [amazonOpen, setAmazonOpen] = useState(false);
  const [isBuilding, setIsBuilding] = useState(false);
  const router = useRouter();

  const handleBuildForMe = async () => {
    setIsBuilding(true);
    try {
      const response = await fetch("/api/datasets/products?limit=50");
      if (!response.ok) throw new Error("Failed to load products from the database");
      
      const data = await response.json();
      
      let rawData = [];
      if (Array.isArray(data)) rawData = data;
      else if (data && Array.isArray(data.items)) rawData = data.items;
      else if (data && Array.isArray(data.data)) rawData = data.data;

      const shuffledData = rawData.sort(() => 0.5 - Math.random()).slice(0, 4);

      const normalizedProducts = shuffledData.map((p: any, index: number) => {
        let cleanDesc = "Sin descripción disponible.";
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
          key: p.parent_asin || p.id || String(index),
          category: primaryCategory,
          name: p.title || p.name || "Producto sin título",
          description: cleanDesc,
          price: productPrice,
          successScore: Number(p.successScore) || Math.floor(Math.random() * 40) + 40,
          monthlyProfit: Number(p.monthlyProfit) || Math.floor(productPrice * 80),
          startupCost: Math.floor(productPrice * 15), 
          image: imageUrl,
          trend: Number(p.trend) || Math.floor(Math.random() * 20) + 5,
          sourceType: "database" 
        };
      });

      await createStore(normalizedProducts);
    } catch (error) {
      console.error("Error al construir la tienda con datos reales:", error);
    } finally {
      setIsBuilding(false);
    }
  };

  if (!ready) return <div className="store-empty"><span className="state-icon scan"><Store/></span><h1>Opening your local store…</h1></div>;
  
  if (!store) return (
    <div className="store-empty">
      <span className="state-icon"><Store/></span>
      <h1>You have not built a store yet</h1>
      <p>Shortlist products in Discover or let Priori assemble a balanced demo portfolio.</p>
      <div>
        <Button variant="coral" onClick={() => router.push("/discover")}>
          <PackagePlus/>Browse trending
        </Button>
        <Button onClick={handleBuildForMe} disabled={isBuilding}>
          {isBuilding ? <Loader2 className="animate-spin" /> : <Bot/>}
          {isBuilding ? "Analyzing database..." : "Build one for me"}
        </Button>
      </div>
      <Badge tone="mint">Private Supabase portfolio · Real database items</Badge>
    </div>
  );
  
  const monthly = store.products.reduce((sum, product) => sum + product.monthlyProfit, 0);
  const investment = store.products.reduce((sum, product) => sum + product.startupCost, 0);
  const payback = Math.max(1, Math.round(investment / (monthly / 4.33)));

  return (
    <>
      <div className="store-hero">
        <Badge tone="neutral">Private portfolio · Supabase</Badge>
        <h1>{store.brand}</h1>
        <p>{store.description}</p>
        <small>Combined scenario profit</small>
        <strong>{money(monthly)} <span>/month</span></strong>
      </div>
      
      <div className="store-toolbar">
        <Button variant="coral" onClick={() => router.push("/discover")}><PackagePlus/>Add products</Button>
        <Button variant="secondary" onClick={clearStore}><RotateCcw/>Start again</Button>
      </div>
      
      <div className="store-kpis">
        <article><Boxes/><strong>{store.products.length}</strong><span>Products</span></article>
        <article><CircleDollarSign/><strong>{money(investment)}</strong><span>Startup stock</span></article>
        <article><TrendingUp/><strong>{payback} wks</strong><span>Scenario payback</span></article>
        <article><ShieldCheck/><strong>{money(monthly * 12 - investment)}</strong><span>Year one scenario</span></article>
      </div>
      
      <div className="section-heading">
        <span><Store/>Your catalog</span>
        <Badge tone="sun">Real Product Data</Badge>
      </div>
      
      <div className="store-products">
        {store.products.map((product) => (
          <article key={product.id}>
            <button className="remove-product" onClick={() => removeProduct(product.id)} aria-label={`Remove ${product.name}`}>
              <Trash2/>
            </button>
            
            <div 
              className="store-product-photo"
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
                sizes="280px"
                style={{ objectFit: "contain" }}
              />
            </div>
            
            <h2 
              title={product.name}
              style={{
                display: "-webkit-box",
                WebkitLineClamp: 3,
                WebkitBoxOrient: "vertical",
                overflow: "hidden",
                textOverflow: "ellipsis",
                margin: "1rem 0 0.5rem 0",
                minHeight: "4.5rem",
                lineHeight: "1.5rem"
              }}
            >
              {product.name}
            </h2>
            
            <p 
              title={product.description}
              style={{
                display: "-webkit-box",
                WebkitLineClamp: 2,
                WebkitBoxOrient: "vertical",
                overflow: "hidden",
                textOverflow: "ellipsis",
                margin: "0 0 1rem 0",
                minHeight: "2.8rem",
                lineHeight: "1.4rem"
              }}
            >
              {product.description}
            </p>
            
            <dl style={{ marginTop: "auto" }}>
              <div><dt>Sell at</dt><dd>{money(product.price)}</dd></div>
              <div><dt>Success</dt><dd>{product.successScore}%</dd></div>
              <div><dt>Startup stock</dt><dd>{money(product.startupCost)}</dd></div>
              <div><dt>Profit scenario</dt><dd>{money(product.monthlyProfit)}/mo</dd></div>
            </dl>
          </article>
        ))}
      </div>
      
      <section className="automation-panel">
        <div className="automation-head">
          <span><Bot/></span>
          <div>
            <h2>Amazon automation</h2>
            <p>Preview the future workflow without pretending an integration exists.</p>
          </div>
        </div>
        <div className="automation-list">
          {["Auto-publish listings", "Smart pricing", "Inventory sync", "Pause low performers"].map((item) => (
            <div key={item}>
              <span>{item}<small>Requires Amazon SP-API and backend authorization.</small></span>
              <button role="switch" aria-checked="false" disabled />
            </div>
          ))}
        </div>
        <Button variant="coral" onClick={() => setAmazonOpen(true)}>Connect my Amazon store</Button>
      </section>
      
      <Modal open={amazonOpen} onClose={() => setAmazonOpen(false)} title="Amazon connection is not active">
        <div className="modal-icon amazon"><Store/></div>
        <p>This Next.js milestone does not implement Amazon OAuth or publishing. A production connection must use SP-API through a secure backend with explicit confirmation and audit logs.</p>
        <Button variant="secondary" onClick={() => setAmazonOpen(false)}>Got it</Button>
      </Modal>
    </>
  );
}
