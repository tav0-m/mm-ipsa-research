# Contribuir

1. Cree una rama para el cambio.
2. No incluya archivos bajo `outputs/`, precios ni retornos por activo.
3. Añada pruebas para cualquier contrato matemático o temporal nuevo.
4. Ejecute:

```powershell
.\scripts\release_check.ps1
```

5. Para cambios que afecten resultados, regenere el pipeline y ejecute:

```powershell
.\scripts\release_check.ps1 -Full
```

Una mejora de software no debe presentarse como mejora financiera sin evaluación OOS, incertidumbre y comparación con benchmarks simples.
