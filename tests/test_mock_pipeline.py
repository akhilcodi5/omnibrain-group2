import asyncio
from app.api.routes_chat import query_agent_orchestrator, ChatQueryRequest

async def main():
    req = ChatQueryRequest(
        query="Extract the 'Net sales by reportable segment' table. Compute the year-over-year percentage change for iPhone and Services revenue between 2022 and 2023. Did Services growth offset the decline in iPhone sales?",
        pdf_name="Apple 10-K PDF 2023",
        referenced_images=[
            "chart_Apple 10-K PDF 2023_p3_1_c49ec9.png",
            "chart_Apple 10-K PDF 2023_p21_1_15b779.png",
            "chart_Apple 10-K PDF 2023_p22_1_dd0b8c.png",
            "chart_Apple 10-K PDF 2023_p24_1_9f14e8.png",
            "table_Apple 10-K PDF 2023_p24_1_03efef.png"
        ]
    )
    resp = await query_agent_orchestrator(req)
    print("FINAL RESPONSE:")
    print("=" * 80)
    print(resp.final_response)
    print("=" * 80)
    print("VISUAL EVIDENCE COUNT:", len(resp.visual_evidence))

if __name__ == "__main__":
    asyncio.run(main())
