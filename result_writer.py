import json

class ResultWriter:

    @staticmethod
    def save_results(results, results_df, output_file="chatbot_results.json"):
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump({
                "total_tests": len(results),
                "results": results,
                "summary": {
                    "avg_similarity": float(results_df["similarity"].mean())
                }
            }, f, ensure_ascii=False, indent=2)

        print(f"\n✓ Results saved to '{output_file}'")