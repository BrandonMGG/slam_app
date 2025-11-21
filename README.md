## Descripción general del sistema SLAM monocular 2D en Android

Este repositorio contiene la implementación de un algoritmo SLAM monocular 2D para Android diseñado para ejecutarse directamente en un teléfono, bajo un presupuesto de tiempo estricto por cuadro, y generar trayectorias 2D coherentes en interiores y exteriores. El proyecto forma parte de un Trabajo Final de Graduación y combina tres ideas principales:

1. Un front-end de odometría visual por características (ORB + RANSAC).
2. Una fusión visual–inercial ligera mediante un filtro de Kalman extendido (EKF) de rumbo y sesgo.
3. Un ajustador adaptativo tipo bandit (UCB1) que selecciona perfiles discretos de operación según la calidad interna de la estimación.

En la práctica, el sistema:

- Toma vídeo de la cámara trasera del teléfono y lo procesa cuadro a cuadro.
- Extrae puntos clave ORB y los empareja entre cuadros usando descriptores binarios.
- Estima el movimiento 2D (traslación en un plano y ángulo de giro) con modelos geométricos robustos (esencial/homografía con RANSAC).
- Fusiona la información de cámara e IMU con un EKF ligero para:
  - Estabilizar el rumbo y reducir zigzags numéricos en la trayectoria.
  - Aplicar un ZUPT en reposo para evitar integrar movimiento ficticio cuando el teléfono está quieto.
- Ajusta automáticamente parámetros del módulo de odometría visual (umbral de ratio test, escala de RANSAC, modo ORB normal/fast, etc.) mediante un bandit UCB1 que:
  - Elige entre perfiles discretos (N0–N2, F0–F2) según un contexto de movimiento (normal o fast).
  - Usa como señal de recompensa métricas internas como inlier_ratio, número de matches, fallos de VO y cambios de brazo.
  - Impone permanencias mínimas y cooldown para evitar cambios erráticos de configuración entre cuadros.

Todo esto se realiza respetando un presupuesto aproximado de 50 ms por cuadro (alrededor de 20 Hz). El pipeline está diseñado para mantener una cadencia estable, con latencias acotadas y sin depender de un back-end pesado continuo; no se ejecuta bundle adjustment ni cierre de lazo en cada cuadro, lo que hace viable el sistema en un teléfono Android.

Además, la aplicación:

- Registra logs detallados por cuadro en el dispositivo Android, incluyendo:
  - Métricas de odometría visual (puntos clave, matches, inliers, paralaje, causas de rechazo).
  - Estado del EKF (yaw, sesgo, confianza de IMU, detección de reposo).
  - Decisiones del bandit (brazo seleccionado, contexto, recompensa, cambios de modo ORB).
  - Datos de rendimiento en Android (tasa de cuadros procesados, uso de CPU, memoria, temperatura, batería).
- Permite analizar posteriormente las rutas y las métricas usando scripts de apoyo en PC, por ejemplo para generar figuras y tablas como las del documento final del TFG.

## Cómo ejecutar la aplicación en Android

Esta sección describe cómo clonar el repositorio, preparar el entorno de compilación y generar el APK para instalarlo en un dispositivo Android.

### 1. Requisitos previos

El proyecto está pensado para compilarse con Buildozer sobre Linux. En Windows se recomienda usar WSL2 o una máquina virtual con Ubuntu.

Antes de comenzar se debe de tener:

- Sistema operativo: Ubuntu 20.04 o superior (o equivalente).
- Python 3.10 o 3.11 instalado.
- Git instalado.
- Herramientas de compilación: gcc, make, unzip, etc.
- Java JDK (8 u 11).
- Dispositivo Android con:
  - Opciones de desarrollador activadas.
  - Depuración por USB activada.

En Ubuntu, paquetes:

```bash
sudo apt update
sudo apt install -y git python3 python3-venv python3-pip \
    openjdk-11-jdk build-essential unzip zip \
    libncurses5 libtinfo5 zlib1g-dev android-tools-adb
```
Después, se necesita crear y activar un entorno virtual de Python e instala Buildozer:

```bash
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install buildozer cython
```

### 2. Clonar el repositorio y preparar el entorno
Clona este repositorio y entra en la carpeta del proyecto:   
```bash
git clone https://github.com/BrandonMGG/slam_app.git
cd slam_app
source venv/bin/activate
```


###  3. Conectar el dispositivo Android

En el teléfono Android, habilitar las opciones de desarrollador.
  
  Activa la depuración por USB.
  
  Conecta el dispositivo al PC mediante cable USB.
  
  Comprueba que el sistema lo reconoce:
```bash
  adb devices
```
O inalambricamente: 

  Opcion de Depuración Inalambrica
  Vincular Dispositivo con un codigo
  En la consola del PC:
  ```bash
  adb pair ip:puerto
  adb connect ip:puerto
  ```
### 4. Compilar el APK con Buildozer

Desde la raíz del proyecto (buildozer.spec), ejecutar:
```bash
buildozer -v android debug
```

### 5. Instalar y ejecutar la aplicación en el dispositivo

Con el dispositivo conectado y reconocido por adb, se puede compilar, instalar y ejecutar la aplicación en un solo paso:
```bash
buildozer -v android debug deploy run
```


## Arquitectura del sistema

El sistema se ejecuta íntegramente en el teléfono Android. La aplicación captura fotogramas de la cámara y lecturas inerciales (giroscopio y acelerómetro), estima una trayectoria 2D en vista superior y registra métricas por fotograma para análisis posterior. Todo el procesamiento es local y el lazo de estimación trabaja con una cadencia limitada de forma explícita (del orden de 20 Hz) para respetar el presupuesto temporal y térmico del dispositivo.

### Visión general

A nivel de contexto, el usuario interactúa con una aplicación Android que coordina tres elementos principales: la cámara, los sensores inerciales y el almacenamiento local. La app inicia y detiene la captura, muestra la trayectoria estimada en tiempo real y permite exportar los resultados de cada sesión (logs y figuras).

![Diagrama de contexto del sistema](https://github.com/user-attachments/assets/c15be013-c60f-4094-ad2c-d625dc6837cb)


Diagrama de contexto del sistema (usuario, sensores, aplicación y almacenamiento local)

### Bloques principales

Internamente, la arquitectura se organiza en bloques funcionales que separan captura, estimación y registro de resultados. Esta separación permite mantener el camino crítico libre de bloqueos y facilita la depuración en dispositivo.

![Bloques del núcleo y flujos](https://github.com/user-attachments/assets/920dc5a1-0595-44f1-86cf-56eac803db04)


Bloques del núcleo y flujos principales del sistema

Los bloques principales son:

- Interfaz gráfica y runner  
  Muestra la vista de cámara y la trayectoria 2D en tiempo real. Permite iniciar y detener la captura y el procesamiento, y ofrece la opción de guardar los artefactos de la sesión. La interfaz envía solo banderas de control para no bloquear el procesamiento.

- Módulo de captura y sincronización  
  Recibe los fotogramas de la cámara con su marca de tiempo y consolida una ventana de lecturas de IMU alrededor de cada instante. Aplica la rotación discreta correspondiente a la orientación física del dispositivo y entrega al núcleo imágenes listas para procesar.

- Módulo de odometría visual (VO)  
  Ejecuta detección y descripción ORB, emparejamiento binario por distancia de Hamming y estimación robusta de la matriz esencial mediante RANSAC. A partir de los inliers y el paralaje decide aceptar o rechazar el incremento de pose. Si la calidad es baja, conserva el último estado válido para evitar saltos.

- Módulo de fusión inercial ligera  
  Recibe la rotación estimada por VO y la combina con la señal de la IMU mediante un filtro ligero centrado en el rumbo (yaw) y el sesgo. Aplica actualizaciones de velocidad cero (ZUPT) cuando el sistema está en reposo y limita la contribución inercial si hay desincronización entre cámara e IMU.

- Módulo de generación de trayectoria 2D  
  Proyecta la pose estabilizada en un plano fijo y actualiza la polilínea que representa la ruta estimada. Esta trayectoria se envía a la interfaz para su visualización y se registra para análisis fuera de línea.

- Módulo de observabilidad y resultados  
  Registra métricas por fotograma (fps, inliers, paralaje, banderas de aceptación y causas de rechazo, entre otras) y genera, al final de la sesión, un log detallado y un resumen con estadísticas agregadas. La escritura se hace fuera del hilo crítico.

- Módulo de ajuste adaptativo (bandido multi-brazo)  
  Consume métricas agregadas en ventanas de tiempo y selecciona, mediante una política tipo UCB1, un perfil discreto de parámetros (por ejemplo, umbrales de VO y peso de la fusión inercial). Las decisiones se aplican a baja frecuencia y con permanencia mínima para evitar oscilaciones de configuración.

### Flujo de datos por fotograma

El flujo por fotograma sigue siempre la misma ruta: captura, estimación geométrica, estabilización, trayectoria y registro. Este diseño reduce la complejidad del camino crítico y mantiene la latencia controlada en el dispositivo móvil.

![Diagrama de flujo por fotograma](https://github.com/user-attachments/assets/3d8f26c4-a074-4a7a-9ca5-1202de604530)


Flujo por fotograma desde la captura hasta la actualización de la trayectoria

De forma resumida, el ciclo de cada fotograma es:

1. La cámara entrega un fotograma en escala de grises con su marca de tiempo.
2. El proveedor de IMU entrega una ventana de lecturas de giroscopio y acelerómetro centrada en ese instante.
3. El módulo de odometría visual ejecuta ORB, emparejamiento y RANSAC para estimar la pose relativa y las métricas de calidad.
4. Se decide aceptar o rechazar el incremento de pose según inliers y paralaje. Si se rechaza, se mantiene el último estado válido.
5. Si se acepta, la fusión inercial ligera estabiliza el rumbo con la ventana inercial asociada y aplica, cuando corresponde, ZUPT en reposo.
6. La pose estabilizada actualiza la trayectoria 2D, que se dibuja en la interfaz.
7. Se registran métricas por fotograma y, cada cierto número de cuadros, el módulo de bandido actualiza el perfil de parámetros según las recompensas internas.

Este esquema mantiene un lazo visual-inercial ligero, ajustable mediante perfiles discretos y con registro suficiente para analizar el comportamiento del sistema en escenarios de interior y exterior.



## Resultados destacados

Esta sección resume el comportamiento del sistema en un teléfono Android real, tanto en recorridos interiores como exteriores, y muestra el impacto de la fusión visual–inercial ligera y del ajustador UCB1 sobre perfiles discretos. No se incluyen todas las tablas del documento de tesis, solo los resultados más relevantes para entender cómo se comporta el sistema en la práctica.

A continuacion se presentan las figuras, que representan tanto los recorridos téoricos como los prácticos:


![Ruta teórica interior Int1](https://github.com/user-attachments/assets/f6f78816-dd5a-421c-826e-d1f6e72157f1)

Ruta teórica interior Int1

![Trayectorias prácticas interior Int1 (3 ejecuciones)](https://github.com/user-attachments/assets/ddb4891e-dcef-404c-a03a-767aafb4d2dd)

Trayectorias prácticas interior Int1 (3 ejecuciones)


![Ruta teórica exterior Ext1](https://github.com/user-attachments/assets/a20ceb3d-a27c-4090-8796-3ebb8a085f64)

Ruta teórica exterior Ext1


![Trayectorias prácticas exterior Ext1 (3 ejecuciones)](https://github.com/user-attachments/assets/62a6199c-dfaf-4b93-aaf3-05629086a085)

Trayectorias prácticas exterior Ext1 (3 ejecuciones)



### Escenarios de prueba

Se evaluaron dos tipos de recorridos:

- Int1: ruta interior en pasillos de una vivienda, con iluminación difusa y tramos de baja textura (paredes lisas, superficies uniformes).
- Ext1: ruta exterior más larga, recorrida en bicicleta, sosteniendo el teléfono con una mano, lo que introduce vibraciones adicionales y giros más bruscos.

Para cada recorrido se realizaron tres ejecuciones con la misma configuración. Cada ejecución genera:

- Una trayectoria 2D en el plano.
- Un archivo de métricas por fotograma y un resumen por sesión.

En todos los experimentos:

- Resolución fija de cámara: 640×480.
- Lazo de procesamiento limitado explícitamente a una cadencia objetivo cercana a 20 Hz.
- Mismo dispositivo Android y misma configuración del pipeline.

### Prueba interior (Int1)

En interior se buscó comprobar si el sistema se mantiene estable en un entorno difícil para un VO monocular: poca textura, pasillos estrechos y giros frecuentes.

Resumen de comportamiento en Int1 con el ajustador bandit activado:

- Tasa de aceptación por sesión entre 84,5 % y 88,8 %.
- Razón de inliers mediana muy estable, entre 0,783 y 0,794.
- Paralaje mediano por cuadro entre 6,6 y 8,5 píxeles, reflejando giros y cambios de dirección a corta distancia.
- Puntos clave y matches por cuadro prácticamente constantes (medianas de 1600 y 800).
- Distancia total acumulada en unidades de VO (uVO) en el rango 233–263 uVO, usada como referencia relativa entre ejecuciones.
- Cero reinicios de VO en las tres corridas.
- El EKF de rumbo y sesgo se actualiza prácticamente una vez por cada cuadro aceptado.
- Cuadros marcados como reposo (detección de is_stationary y ZUPT) en torno al 1 % del recorrido, coherente con un movimiento mayoritariamente continuo.

Respecto al ajustador UCB1:

- El contexto normal domina casi toda la ruta; el contexto fast se activa solo de forma puntual.
- Los brazos N1 y N2 se usan la mayor parte del tiempo, con pocos cambios de brazo y pocos cambios de modo ORB.
- La recompensa mediana del bandit se mantiene alrededor de 0,76, señal de que el sistema converge hacia perfiles estables para este entorno.

Las trayectorias prácticas de Int1 reproducen la secuencia de tramos rectos y giros prevista por la ruta teórica. Las diferencias entre ejecuciones son coherentes con variaciones ligeras de paralaje y aceptación, pero sin fallos catastróficos ni reinicios.

### Prueba exterior (Ext1)

En exterior la ruta es más larga y se recorre en bicicleta, sosteniendo el teléfono con una mano, lo que genera vibraciones y cambios de orientación más bruscos.

Resumen de comportamiento en Ext1:

- Tasa de aceptación por sesión entre 89,4 % y 95,0 %. Una de las ejecuciones alcanza claramente los mejores valores.
- Razón de inliers mediana en el rango 0,781–0,792, muy próxima a la de interior.
- Paralaje mediano por cuadro entre 3,4 y 4,3 píxeles, menor que en interior debido a referencias más lejanas y tramos más rectos.
- Puntos clave y matches por cuadro con las mismas medianas que en interior (1600 y 800).
- Distancia acumulada en uVO mucho mayor, en torno a 773–839 uVO, consistente con recorridos más largos y continuos.
- Cero reinicios de VO en las tres corridas.
- El EKF vuelve a actualizarse casi cuadro a cuadro aceptado, sin colas ni bloqueos.
- No se registran cuadros en reposo, ya que el movimiento es prácticamente continuo en bicicleta.

Respecto al ajustador:

- El contexto fast se activa con mucha más frecuencia que en interior, alrededor de un 15 % de los cuadros.
- La distribución de brazos combina N1, N2 y los perfiles rápidos F0–F2, que se usan más cuando la ruta es dinámica.
- La recompensa mediana del bandit se sitúa entre 0,74 y 0,77, con la mejor combinación de aceptación y paralaje en la ejecución con mayor uso de perfiles rápidos y contexto fast.

Las trayectorias prácticas de Ext1 siguen el patrón de tramos rectos y giros planificado en la ruta teórica, con giros más marcados en la ejecución que presenta mayor paralaje y mayor tasa de aceptación.

### Comparación interior vs exterior

Comparando las seis ejecuciones (tres en Int1 y tres en Ext1) bajo la misma cadencia objetivo y el mismo dispositivo:

- Cadencia efectiva:
  - Interior: fps mediano procesado alrededor de 8,1 Hz.
  - Exterior: fps mediano procesado alrededor de 9,0 Hz.
  - La diferencia, de casi 1 Hz, indica que en exterior se procesan ligeramente más cuadros útiles, pese a la mayor vibración.

- Aceptación e inliers:
  - Aceptación media en interior alrededor de 86,9 %.
  - Aceptación media en exterior alrededor de 91,3 %.
  - La razón de inliers mediana es casi idéntica en ambos escenarios, alrededor de 0,79, lo que indica un soporte geométrico muy consistente.

- Paralaje:
  - Interior: paralaje mediano promedio de aproximadamente 7,2 píxeles por cuadro.
  - Exterior: paralaje mediano promedio de aproximadamente 3,8 píxeles por cuadro.
  - En interior se observa más paralaje por cuadro por los giros cercanos; en exterior, aunque el paralaje instantáneo es menor, el entorno ofrece más textura útil y se compensa con mayor aceptación.

- Rechazos y reposo:
  - Los rechazos por baja coincidencia o poco paralaje (E01) son más frecuentes en interior que en exterior.
  - En interior aparecen cuadros marcados como reposo en torno al 1 %; en exterior casi no hay pausas.

- Política adaptativa:
  - En interior domina el contexto normal y los brazos N1 y N2; los perfiles rápidos casi no se usan.
  - En exterior aumentan tanto el contexto fast como los brazos F0–F2, y se observan más cambios de modo ORB, reflejando un entorno más exigente en términos de dinámica.

En conjunto, interior ofrece más paralaje por cuadro pero algo menos de aceptación, mientras que exterior consigue más cuadros válidos en recorridos más largos y dinámicos, manteniendo estabilidad y sin reinicios.

### Impacto del ajustador UCB1 (con vs sin bandit)

Para aislar el efecto del bandit, se repitió la ruta interior Int1 con el ajustador desactivado y se comparó con las ejecuciones equivalentes con UCB1 activado.

Resultados agregados:

- Aceptación promedio:
  - Sin bandit: 83,72 %.
  - Con bandit: 86,90 %.
  - Mejora absoluta: aproximadamente 3,18 puntos porcentuales.

- Calidad geométrica:
  - Razón de inliers media: de 0,611 a 0,688, con una mejora relativa de alrededor del 12,6 %.
  - Razón de inliers mediana: de 0,707 a 0,790, con mejora relativa cercana al 11,7 %.
  - Matches medianos por cuadro: de 749,5 a 800, con una mejora de alrededor del 6,7 %.

- Inliers efectivos por cuadro (inlier_ratio_med multiplicado por matches_med):
  - Sin bandit: 529,7.
  - Con bandit: 632,0.
  - Mejora relativa de aproximadamente 19,3 %.

- Rechazos y estabilidad:
  - Rechazos E01 por cuadro: de 15,41 % a 12,88 %, reducción de 2,52 puntos porcentuales.
  - Reinicios de VO: una serie con 1 reinicio sin bandit frente a 0 reinicios con bandit.

Este aumento en la probabilidad de inlier reduce el número de iteraciones que necesita RANSAC para alcanzar una confianza alta, lo que se traduce en modelos geométricos que convergen antes y con menor varianza en la estimación de pose, manteniendo la misma cadencia de procesamiento.

En resumen, el bandit UCB1 selecciona configuraciones de umbrales y perfiles que aumentan el soporte geométrico efectivo incluso sin maximizar el paralaje en cada cuadro, reduciendo rechazos y evitando reinicios del VO.

### Aportes de la IMU al pipeline visual

La IMU se integra mediante un filtro de Kalman extendido ligero de rumbo y sesgo, con detección de reposo tipo ZUPT. A partir de las métricas derivadas en interior se observan los siguientes aportes:

- Aceptación con bajo paralaje:
  - A@P<4 píxeles aproximadamente igual a 56,9 %. El sistema sigue aceptando alrededor de la mitad de los cuadros incluso cuando el paralaje instantáneo es bajo.

- Estabilidad de rumbo y rectitud:
  - Deriva de rumbo por unidad de recorrido en tramos rectos |Δyaw|/Δdist de alrededor de 0,066 grados por unidad de VO.
  - Rectitud en segmentos rectos (RMS) de aproximadamente 0,021 unidades de VO.
  - Estos valores indican que el rumbo se mantiene estable y se evitan trayectorias con curvatura artificial o zigzag numérico.

- Reposo y ZUPT:
  - Velocidad residual durante reposo prácticamente nula.
  - Al detectar reposo, el filtro corrige la velocidad a cero y evita integrar movimiento ficticio.

- Giros rápidos:
  - Razón de inliers durante giros rápidos alrededor de 0,75.
  - Incluso en curvas con altas tasas angulares, el sistema mantiene un nivel de inliers suficiente para actualizar la pose sin cortes.

Estos indicadores explican por qué, pese a usar una fusión inercial ligera y sin un back-end completo, el sistema logra trayectorias continuas y razonablemente estables en escenarios reales.

### Consumo de recursos en Android

Para estimar la carga computacional y energética se instrumentó la ruta exterior Ext1 durante tres ejecuciones consecutivas, midiendo CPU, memoria, temperatura y batería:

- CPU:
  - Mediana entre aproximadamente 115,7 % y 117,0 % en un sistema multinúcleo, equivalente a un núcleo completo más una fracción de otro.
  - Percentil 95 entre 120,2 % y 122,2 %.
  - Tiempo por encima del 120 % entre 6,9 % y 18,3 % del recorrido, asociado a fases intensivas de extracción y emparejamiento de características o giros rápidos.
  - El cambio de modo ORB tiene un impacto mediano bajo en CPU, por lo que las conmutaciones no penalizan de forma significativa.

- Memoria:
  - PSS mediana entre 197 MB y 242 MB, sin tendencia de crecimiento apreciable.
  - No se observan fugas ni presión sostenida sobre el recolector de basura.

- Batería y temperatura:
  - Descenso de batería de alrededor del 1 % por ejecución de unos 2 minutos, lo que extrapolado da un consumo aproximado de 30–34 % por hora en este escenario de estrés.
  - Temperaturas medianas entre aproximadamente 28,2 °C y 31,5 °C, sin signos claros de reducción de rendimiento por temperatura.

- Registro:
  - Los huecos de muestreo del logger de rendimiento no corresponden a bloqueos del VO, que se mantiene sin reinicios durante las tres ejecuciones.

En conjunto, las pruebas de rendimiento muestran que el pipeline por características con fusión inercial ligera y bandit UCB1 se puede ejecutar de forma sostenida en un teléfono Android moderno, con uso de CPU alto pero estable, memoria controlada y temperaturas moderadas, mientras mantiene trayectorias coherentes y sin reinicios en recorridos prolongados.


