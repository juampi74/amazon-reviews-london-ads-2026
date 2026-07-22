import { NextResponse } from "next/server";
import { analysisRequestSchema, fastApiAnalysisSchema } from "@/lib/api/schema";
import { authenticatedBackendRequest } from "@/lib/api/server-proxy";

export async function POST(request: Request) {
  const parsed = analysisRequestSchema.safeParse(await request.json().catch(() => null));
  if (!parsed.success) {
    return NextResponse.json({
      error: "Check the product details and try again.",
      fields: parsed.error.flatten().fieldErrors,
    }, { status: 422 });
  }

  try {
    const response = await authenticatedBackendRequest("/v1/analyses", {
      method: "POST",
      body: JSON.stringify({ ...parsed.data, request_id: parsed.data.request_id ?? crypto.randomUUID() }),
    });

    const body = await response.json().catch(() => null);

    if (!response.ok) {
      return NextResponse.json({
        error: body?.detail ?? body?.error ?? "The model rejected these product details.",
      }, { status: response.status });
    }

    const model = fastApiAnalysisSchema.safeParse(body);
    if (!model.success) {
      console.error("Zod Schema Error:", JSON.stringify(model.error.flatten(), null, 2));
      console.error("Body recibido de FastAPI:", JSON.stringify(body, null, 2));

      return NextResponse.json({ error: "The backend returned an unexpected response." }, { status: 502 });
    }

    return NextResponse.json(model.data);
  } catch (error) {
    const timeout = error instanceof Error && error.name === "TimeoutError";
    return NextResponse.json({ 
      error: timeout ? "The model took too long to respond." : "The backend is unavailable." 
    }, { status: 503 });
  }
}