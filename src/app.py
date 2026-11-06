"""Ejecuta el caso completo y guarda modelos, figuras y la evaluación."""

from housing_clusters import run_pipeline


def main() -> None:
    result = run_pipeline()
    print(result["lectura_escala"])
    print()
    print(result["lectura_k"])
    print()
    print(result["lectura_clusters"])
    print()
    print(result["lectura_modelo"])
    print()
    print("modelo_supervisado", result["selected_name"])
    print("accuracy_test", f"{result['metrics']['accuracy']:.4f}")
    print("f1_macro_test", f"{result['metrics']['f1_macro']:.4f}")
    print("reload_kmeans", result["reload_kmeans_ok"])
    print("reload_supervised", result["reload_supervised_ok"])


if __name__ == "__main__":
    main()
