"use client";

import { useCallback, useEffect, useState } from "react";
import { createClient } from "@/lib/supabase/client";
import { demoStoreSchema, storeStateSchema } from "@/lib/api/schema";
import type { DemoStore, StoreProduct, StoreState } from "@/lib/api/types";

const cartPrefix = "launchly-cart-v1";
const storePrefix = "launchly-store-v1";

function normalizeProduct(item: any): any {
  if (!item) return item;
  const rawImage = Array.isArray(item.images) ? item.images[0]?.image_url ?? item.images[0] : item.image;
  const rawCategory = Array.isArray(item.categories) 
    ? (item.categories[item.categories.length - 1]?.category_name ?? item.categories[item.categories.length - 1] ?? "General")
    : (item.category ?? "General");

  const productPrice = Number(item.price) || 25;
  const monthlyProfit = Number(item.monthlyProfit) || Math.floor(productPrice * 80);
  const startupCost = Number(item.startupCost) || Math.floor(productPrice * 15);
  const successScore = Number(item.successScore) || Math.floor(Math.random() * 40) + 40;

  return {
    ...item,
    id: String(item.id ?? item.parent_asin ?? crypto.randomUUID()),
    title: item.title ?? item.name ?? "Product",
    name: item.name ?? item.title ?? "Product",
    price: productPrice,
    monthlyProfit: monthlyProfit,
    startupCost: startupCost,
    successScore: successScore,
    image: typeof rawImage === "string" ? rawImage : "https://images.unsplash.com/photo-1523381210434-271e8be1f52b",
    category: typeof rawCategory === "string" ? rawCategory : "General",
  };
}

function readLocal(identity: string): StoreState {
  try {
    const shortlistRaw = JSON.parse(localStorage.getItem(`${cartPrefix}:${identity}`) ?? "[]");
    const storeRaw = JSON.parse(localStorage.getItem(`${storePrefix}:${identity}`) ?? "null");
    const shortlist = Array.isArray(shortlistRaw)
      ? shortlistRaw.map((item) => normalizeProduct(item))
      : [];
    const parsedStore = demoStoreSchema.safeParse(storeRaw);
    return { store: parsedStore.success ? parsedStore.data : storeRaw, shortlist };
  } catch {
    return { store: null, shortlist: [] };
  }
}

function saveLocal(identity: string, store: DemoStore | null, shortlist: StoreProduct[]) {
  try {
    localStorage.setItem(`${cartPrefix}:${identity}`, JSON.stringify(shortlist));
    localStorage.setItem(`${storePrefix}:${identity}`, JSON.stringify(store));
  } catch {}
}

function clearLocal(identity: string) {
  localStorage.removeItem(`${cartPrefix}:${identity}`);
  localStorage.removeItem(`${storePrefix}:${identity}`);
}

async function persistState(store: DemoStore | null, shortlist: StoreProduct[]) {
  const normalizedShortlist = shortlist.map((item) => normalizeProduct(item));
  const normalizedStore = store
    ? {
        ...store,
        products: store.products.map((item) => normalizeProduct(item)),
      }
    : null;

  try {
    const response = await fetch("/api/store/import", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ request_id: crypto.randomUUID(), store: normalizedStore, shortlist: normalizedShortlist }),
    });
    
    const body = await response.json().catch(() => null);
    if (response.ok) {
      const parsed = storeStateSchema.safeParse(body);
      if (parsed.success) {
        return parsed.data;
      }
    }
  } catch {}

  return {
    store: normalizedStore,
    shortlist: normalizedShortlist,
  };
}

export function useStorePortfolio() {
  const [identity, setIdentity] = useState("session");
  const [cart, setCart] = useState<StoreProduct[]>([]);
  const [store, setStore] = useState<DemoStore | null>(null);
  const [ready, setReady] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let active = true;
    (async () => {
      let resolvedIdentity = "session";
      try {
        const { data } = await createClient().auth.getClaims();
        resolvedIdentity = String(data?.claims?.sub ?? "session");
        if (!active) return;
        setIdentity(resolvedIdentity);
        const local = readLocal(resolvedIdentity);
        
        let next = { store: local.store, shortlist: local.shortlist };
        try {
          const response = await fetch("/api/store", { cache: "no-store" });
          const remoteBody = await response.json().catch(() => null);
          if (response.ok && remoteBody) {
            next = {
              store: remoteBody.store ?? local.store,
              shortlist: Array.isArray(remoteBody.shortlist) ? remoteBody.shortlist.map(normalizeProduct) : local.shortlist,
            };
          }
        } catch {}

        if (!active) return;
        setStore(next.store);
        setCart(next.shortlist);
      } catch (reason) {
        if (!active) return;
        const local = readLocal(resolvedIdentity);
        setStore(local.store);
        setCart(local.shortlist);
        setError(reason instanceof Error ? reason.message : "The saved store is unavailable.");
      } finally {
        if (active) setReady(true);
      }
    })();
    return () => { active = false; };
  }, []);

  const sync = useCallback(async (nextStore: DemoStore | null, nextCart: StoreProduct[]) => {
    const previousStore = store;
    const previousCart = cart;
    setStore(nextStore);
    setCart(nextCart);
    setError(null);

    saveLocal(identity, nextStore, nextCart);

    try {
      const saved = await persistState(nextStore, nextCart);
      setStore(saved.store);
      setCart(saved.shortlist);
      saveLocal(identity, saved.store, saved.shortlist);
      return saved;
    } catch (reason) {
      setStore(previousStore);
      setCart(previousCart);
      saveLocal(identity, previousStore, previousCart);
      setError(reason instanceof Error ? reason.message : "The store could not be saved.");
      throw reason;
    }
  }, [cart, identity, store]);

  const toggleProduct = useCallback(async (product: StoreProduct) => {
    const cleanProduct = normalizeProduct(product);
    const productId = cleanProduct.id;
    const next = cart.some((item) => String(normalizeProduct(item).id) === String(productId))
      ? cart.filter((item) => String(normalizeProduct(item).id) !== String(productId))
      : [...cart, cleanProduct];
    await sync(store, next);
  }, [cart, store, sync]);

  const createStore = useCallback(async (products = cart) => {
    if (!products.length) return null;
    const normalizedProducts = products.map((p) => normalizeProduct(p));
    const categories = [...new Set(normalizedProducts.map((product) => product.category ?? "General"))];
    
    const prefixes = ["Glow", "Velvet", "Aura", "Dewy", "Lumina", "Silk", "Botanica", "Nova", "Pure", "Luna", "Fleur", "Oasis"];
    const suffixes = ["Rituals", "& Bloom", "Beauty Co.", "Collective", "Cosmetics", "Lab", "Studio", "Naturals", "Skincare", "Aesthetics"];
    
    const randomPrefix = prefixes[Math.floor(Math.random() * prefixes.length)];
    const randomSuffix = suffixes[Math.floor(Math.random() * suffixes.length)];
    const generatedBrand = `${randomPrefix} ${randomSuffix}`;

    const next: DemoStore = {
      brand: generatedBrand,
      description: `A focused, data-picked beauty collection spanning ${categories.join(", ")}. Built to test demand before committing serious capital.`,
      currency: "USD",
      products: normalizedProducts,
    };
    
    await sync(next, normalizedProducts);
    return next;
  }, [cart, sync]);

  const removeProduct = useCallback(async (id: number | string) => {
    if (!store) return;
    const products = store.products.filter((product) => String(normalizeProduct(product).id) !== String(id));
    const nextCart = cart.filter((product) => String(normalizeProduct(product).id) !== String(id));
    await sync(products.length ? { ...store, products } : null, nextCart);
  }, [cart, store, sync]);

  const clearStore = useCallback(async () => {
    try {
      await fetch("/api/store", { method: "DELETE" });
    } catch {}
    setStore(null);
    setCart([]);
    clearLocal(identity);
  }, [identity]);

  return { cart, store, ready, error, toggleProduct, createStore, removeProduct, clearStore };
}
