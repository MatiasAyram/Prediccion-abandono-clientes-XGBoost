# Predicción de Abandono de Clientes con XGBoost y SMOTE

Este proyecto implementa un sistema de predicción de abandono de clientes utilizando XGBoost, la técnica de sobremuestreo SMOTE para balancear clases, y herramientas de evaluación como la optimización de umbral, reporte de clasificación y matriz de confusión.

## Características del modelo

- **Algoritmo**: XGBoost (`xgb.train` con early stopping)
- **Dataset**: [Telco Customer Churn](https://www.kaggle.com/blastchar/telco-customer-churn)
- **Incluye**:
  - SMOTE para balanceo de clases
  - Cálculo automático de `scale_pos_weight`
  - Optimización del umbral según F1-score
  - Early stopping para evitar sobreajuste
  - Codificación de etiquetas (Label Encoding) y escalado (StandardScaler)

## Flujo de trabajo del proyecto
1. **Carga de datos**  
   Se importa el dataset Telco Customer Churn y se realiza una limpieza básica.
2. **Preprocesamiento**  
   - Se codifican variables categóricas con `LabelEncoder`.
   - Se escalan las variables numéricas con `StandardScaler`.
3. **Balanceo de clases**  
   Dado que el conjunto de datos está desbalanceado (pocos casos de abandono), se aplica **SMOTE** para generar datos sintéticos de la clase minoritaria.
4. **Cálculo de `scale_pos_weight`**  
   Se calcula automáticamente según la proporción de clases original, y se pasa como parámetro a XGBoost para mejorar el rendimiento en clasificación desbalanceada.
5. **Entrenamiento del modelo**  
   Se utiliza `xgb.train` (API de bajo nivel) con:
   - `early_stopping` para evitar sobreajuste
   - set de validación
   - métrica `logloss`
6. **Predicción y ajuste de umbral**  
   Se genera una curva de `precision-recall` y se encuentra el **mejor umbral** de probabilidad que maximiza el F1-score.
7. **Evaluación del modelo**  
   - Se genera el reporte de clasificación: precisión, recall, F1.
   - Se grafica la **matriz de confusión** con `Seaborn`.

## Instalación
para clonar el repositorio y preparar el entorno ejecutar en consola:
```bash
git clone https://github.com/MatiasAyram/Prediccion-abandono-clientes-XGBoost.git
cd prediccion-abandono-clientes-XGBoost
pip install -r requirements.txt
```
(El código fue desarrollado en un entorno virtual de Anaconda Navigator(Python 3.10) donde se instalaron las dependencias necesarias.)

## Instalación del dataset
El dataset utilizado es el clásico **Telco Customer Churn Dataset**, disponible en Kaggle:

```markdown
[https://www.kaggle.com/blastchar/telco-customer-churn](https://www.kaggle.com/blastchar/telco-customer-churn)
```
El dataset contiene información detallada sobre más de 7.000 clientes de una empresa de telecomunicaciones ficticia. Incluye datos personales, el tipo de servicios que tienen contratados (como internet, telefonía, soporte técnico, etc.), información sobre sus métodos de pago y facturación, y variables contractuales como la duración del contrato. Además, proporciona indicadores financieros como los cargos mensuales y totales acumulados por cliente.
(si desea la version del dataset con el que ejecutado el codigo puede usar la que se encuentra en el repositorio)

## Resultados
Después de haber tratado, escalado, balanceado el conjunto de datos de entrenamiento y realizado la predicción, el modelo genera un reporte como el siguiente:
Reporte de clasificación: (threshold=0.527)
```text
               precision    recall  f1-score   support

           0       0.90      0.80      0.84      1035
           1       0.57      0.75      0.65       374
    accuracy                           0.78      1409
   macro avg       0.73      0.77      0.75      1409
weighted avg       0.81      0.78      0.79      1409
```
Las métricas obtenidas muestran que el modelo es capaz de predecir correctamente el abandono de clientes en la mayoría de los casos. Sin embargo, se espera que su rendimiento pueda mejorar si se contara con un conjunto de datos más amplio y con características adicionales que aporten mayor información sobre el comportamiento y las preferencias de los usuarios.

## Licencia
Este proyecto esta licenciado bajo la licencia MIT License

## 👨‍💻 Autor

Proyecto realizado por [Matías Ayram](https://github.com/MatiasAyram) como parte de portfolio de proyectos de Data Science / Machine Learning.

---


