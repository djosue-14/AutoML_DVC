# AutoML_DVC
Este es un proyecto Gestión de Experimentación y Modelos con DVC Automated Machine Learning (AutoML)

## Link del repositorio
[Repositorio en Github](https://github.com/djosue-14/AutoML_DVC.git)

## Requisitos

Asegúrate de tener instaladas las siguientes dependencias en tu pc:

- Python 3.x
- DVC
- scikit-learn
- pandas
- xgboost
- joblib
- pyyaml

Puedes instalar las dependencias utilizando pip:

```bash
pip install -r requirements.txt
```

## Instrucciones para Reproducir el Proyecto

1. **Clona el Repositorio**

   Si aún no has clonado el repositorio, utiliza el siguiente comando:

   ```bash
   git clone <URL_DEL_REPOSITORIO>
   cd <NOMBRE_DEL_REPOSITORIO>
   ```

2. **Configuración de DVC**

   Inicializa DVC en el repositorio (si no lo has hecho):

   ```bash
   dvc init
   ```

   Luego, configura tu remota de DVC (por ejemplo, para almacenamiento en S3):

   ```bash
   dvc remote add -d dvc_storage_remote <URL_DEL_REPOSITORIO_REMOTO>
   ```

3. **Cargar Datos Preprocesados**

   Asegúrate de que los datos preprocesados estén disponibles. Si utilizas DVC para manejar los datos, ejecuta:

   ```bash
   dvc pull
   ```

4. **Ejecutar el Pipeline de DVC**

   Una vez que todo esté configurado, puedes ejecutar el pipeline de DVC utilizando el siguiente comando:

   ```bash
   dvc repro
   ```

   Este comando reproducirá todas las etapas del pipeline, incluyendo la generación de modelos y su almacenamiento.

5. **Verificar los Resultados**

   Después de ejecutar el pipeline, verifica que los modelos se hayan guardado correctamente en la carpeta `models/`. Puedes listar los modelos con:

   ```bash
   ls models/
   ```

## Notas

- Asegúrate de que el entrenamiento del modelo se haya completado sin errores.
- Revisa los logs de DVC para más información sobre cualquier fallo en la ejecución del pipeline.
- No olvides cambiar el target y el feature dependiendo del dataset
- Crea la carpeta dvc_storage y results si fuera necesario

