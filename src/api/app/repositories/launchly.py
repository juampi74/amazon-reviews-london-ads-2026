from __future__ import annotations

import base64
from datetime import datetime
import json
from typing import Any
from uuid import UUID

from app.clients.supabase import SupabaseRestClient
from app.schemas.store import DemoStore, StoreImportRequest, StoreProduct, StoreState


def _product_from_row(row: dict[str, Any]) -> StoreProduct:
    payload = dict(row.get("source_payload") or {})
    payload.update({
        "id": payload.get("id", row["source_product_id"]),
        "persistedId": row["id"],
        "key": row["product_key"],
        "category": row["subcategory"],
        "name": row["name"],
        "description": row["description"],
        "price": float(row["selling_price"]),
        "successScore": float(row["success_score"]),
        "monthlyProfit": float(row["estimated_monthly_profit"]),
        "startupCost": float(row["startup_cost"]),
        "image": row.get("image_url"),
        "trend": float(row["trend_pct"]),
        "currency": row["currency"],
        "sourceType": row["source_type"],
    })
    return StoreProduct.model_validate(payload)


class LaunchlyRepository:
    def __init__(self, client: SupabaseRestClient, owner_id: UUID) -> None:
        self.client = client
        self.owner_id = str(owner_id)

    async def persist_analysis(self, request_id: UUID, request: dict[str, Any], result: dict[str, Any]) -> int:
        try:
            value = await self.client.rpc("persist_analysis", {
                "p_request_id": str(request_id), "p_input": request, "p_result": result,
            })
            return int(value) if value is not None else 1
        except Exception as e:
            print(f"Warning: Failed to save the analysis to Supabase (skipping): {e}")
            return 1

    async def list_analyses(self, *, limit: int, cursor: str | None) -> tuple[list[dict[str, Any]], str | None]:
        params: dict[str, Any] = {
            "select": "id,request_id,status,source_type,model_version,dataset_version,created_at,raw_result",
            "owner_id": f"eq.{self.owner_id}", "order": "created_at.desc,id.desc", "limit": limit + 1,
        }
        if cursor:
            try:
                decoded = json.loads(base64.urlsafe_b64decode(cursor.encode()).decode())
                created_at = datetime.fromisoformat(decoded["created_at"])
                row_id = int(decoded["id"])
            except (ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
                raise ValueError("Invalid history cursor.") from exc
            timestamp = created_at.isoformat()
            params["or"] = f"(created_at.lt.{timestamp},and(created_at.eq.{timestamp},id.lt.{row_id}))"
        rows = await self.client.request("GET", "analyses", params=params) or []
        page = rows[:limit]
        next_cursor = None
        if len(rows) > limit and page:
            marker = {"created_at": page[-1]["created_at"], "id": page[-1]["id"]}
            next_cursor = base64.urlsafe_b64encode(json.dumps(marker).encode()).decode()
        return page, next_cursor

    async def get_analysis(self, analysis_id: int) -> dict[str, Any] | None:
        rows = await self.client.request("GET", "analyses", params={
            "select": "id,request_id,status,source_type,model_version,dataset_version,created_at,raw_result",
            "owner_id": f"eq.{self.owner_id}", "id": f"eq.{analysis_id}", "limit": 1,
        }) or []
        return rows[0] if rows else None

    async def import_store(self, payload: StoreImportRequest) -> int | None:
        value = await self.client.rpc("import_store", {
            "p_request_id": str(payload.request_id),
            "p_store": payload.store.model_dump(by_alias=True, exclude_none=True) if payload.store else None,
            "p_shortlist": [item.model_dump(by_alias=True, exclude_none=True) for item in payload.shortlist],
        })
        return int(value) if value is not None else None

    async def get_store_state(self) -> StoreState:
        stores = await self.client.request("GET", "stores", params={
            "select": "id,brand,description,currency", "owner_id": f"eq.{self.owner_id}", "limit": 1,
        }) or []
        products = await self.client.request("GET", "portfolio_products", params={
            "select": "*", "owner_id": f"eq.{self.owner_id}",
        }) or []
        product_map = {int(row["id"]): _product_from_row(row) for row in products}
        shortlist_rows = await self.client.request("GET", "shortlist_items", params={
            "select": "product_id,position", "owner_id": f"eq.{self.owner_id}", "order": "position.asc,id.asc",
        }) or []
        shortlist = [product_map[int(row["product_id"])] for row in shortlist_rows if int(row["product_id"]) in product_map]
        if not stores:
            return StoreState(store=None, shortlist=shortlist)
        store_row = stores[0]
        links = await self.client.request("GET", "store_products", params={
            "select": "product_id,position", "owner_id": f"eq.{self.owner_id}",
            "store_id": f"eq.{store_row['id']}", "order": "position.asc,id.asc",
        }) or []
        store_products = [product_map[int(row["product_id"])] for row in links if int(row["product_id"]) in product_map]
        return StoreState(store=DemoStore(
            brand=store_row["brand"], description=store_row["description"],
            currency=store_row["currency"], products=store_products,
        ), shortlist=shortlist)

    async def delete_store(self) -> None:
        await self.client.request("DELETE", "stores", params={"owner_id": f"eq.{self.owner_id}"}, prefer="return=minimal")
        await self.client.request("DELETE", "shortlist_items", params={"owner_id": f"eq.{self.owner_id}"}, prefer="return=minimal")
        await self.client.request("DELETE", "portfolio_products", params={"owner_id": f"eq.{self.owner_id}"}, prefer="return=minimal")
