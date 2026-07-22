import { NextResponse } from "next/server";
import { authenticatedBackendRequest } from "@/lib/api/server-proxy";

export async function GET(request: Request) {
  try {
    const { searchParams } = new URL(request.url);
    const queryString = searchParams.toString();
    const endpoint = queryString ? `/v1/datasets/products?${queryString}` : "/v1/datasets/products";

    const response = await authenticatedBackendRequest(endpoint, {
      method: "GET",
    });

    const body = await response.json().catch(() => null);

    if (!response.ok) {
      return NextResponse.json(
        { error: body?.detail ?? body?.error ?? "Error al obtener los productos del servidor." },
        { status: response.status }
      );
    }

    return NextResponse.json(body);
  } catch (error) {
    const timeout = error instanceof Error && error.name === "TimeoutError";
    return NextResponse.json(
      { error: timeout ? "El servidor tardó demasiado en responder." : "El backend no está disponible." },
      { status: 503 }
    );
  }
}