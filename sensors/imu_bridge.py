from kivy.utils import platform
from kivy.clock import Clock

IMU_AVAILABLE = False  # flag global para depuración

try:
    if platform == "android":
        from jnius import autoclass, PythonJavaClass, java_method, cast
        PythonActivity = autoclass('org.kivy.android.PythonActivity')
        Context = autoclass('android.content.Context')
        SensorManager = autoclass('android.hardware.SensorManager')
        Sensor = autoclass('android.hardware.Sensor')
    else:
        PythonActivity = Context = SensorManager = Sensor = None
except Exception as e:
    print(f"[IMU] Import/bridge error: {e}")
    PythonActivity = Context = SensorManager = Sensor = None


class _SensorListener(PythonJavaClass):
    __javainterfaces__ = ['android/hardware/SensorEventListener']
    __javacontext__ = 'app'

    def __init__(self, on_sample):
        super().__init__()
        self.on_sample = on_sample

    @java_method('(Landroid/hardware/Sensor;I)V')
    def onAccuracyChanged(self, sensor, accuracy):
        pass

    @java_method('(Landroid/hardware/SensorEvent;)V')
    def onSensorChanged(self, event):
        try:
            stype = event.sensor.getType()
            ts_ns = event.timestamp  # nanosegundos del sensor
            arr = [float(x) for x in event.values]
            self.on_sample(stype, ts_ns, arr)
        except Exception as e:
            print(f"[IMU] onSensorChanged error: {e}")


class IMUBridge:
    """
    Lee acelerómetro y giroscopio y guarda la última muestra.
    Provee get_status() e is_available() para la UI.
    """
    def __init__(self):
        self.mgr = None
        self.listener = None
        self.last_acc = None   # (t_ns, [ax, ay, az])
        self.last_gyro = None  # (t_ns, [gx, gy, gz])
        self.enabled = False

    # --- Compatibilidad con tu main.py ---
    def is_available(self) -> bool:
        return bool(self.enabled)

    @property
    def available(self) -> bool:
        return bool(self.enabled)

    # -------------------------------------

    def _on_sample(self, stype, t_ns, vec):
        if stype == Sensor.TYPE_ACCELEROMETER:
            self.last_acc = (t_ns, vec)
        elif stype == Sensor.TYPE_GYROSCOPE:
            self.last_gyro = (t_ns, vec)

    def start(self):
        global IMU_AVAILABLE
        if platform != "android":
            print("[IMU] No Android platform -> no disponible")
            self.enabled = False
            IMU_AVAILABLE = False
            return False

        try:
            activity = PythonActivity.mActivity
            self.mgr = cast('android.hardware.SensorManager',
                            activity.getSystemService(Context.SENSOR_SERVICE))
            if not self.mgr:
                print("[IMU] SensorManager es None")
                self.enabled = False
                IMU_AVAILABLE = False
                return False

            has_acc = self.mgr.getDefaultSensor(Sensor.TYPE_ACCELEROMETER) is not None
            has_gyro = self.mgr.getDefaultSensor(Sensor.TYPE_GYROSCOPE) is not None
            print(f"[IMU] has_acc={has_acc}  has_gyro={has_gyro}")

            if not (has_acc or has_gyro):
                print("[IMU] No hay sensores requeridos en este dispositivo")
                self.enabled = False
                IMU_AVAILABLE = False
                return False

            self.listener = _SensorListener(self._on_sample)
            rate = SensorManager.SENSOR_DELAY_GAME  # ~50–100 Hz típico

            if has_acc:
                acc = self.mgr.getDefaultSensor(Sensor.TYPE_ACCELEROMETER)
                self.mgr.registerListener(self.listener, acc, rate)
                print("[IMU] Acelerómetro registrado")

            if has_gyro:
                gyro = self.mgr.getDefaultSensor(Sensor.TYPE_GYROSCOPE)
                self.mgr.registerListener(self.listener, gyro, rate)
                print("[IMU] Giroscopio registrado")

            self.enabled = True
            IMU_AVAILABLE = True

            # Comprobación diferida para ver si ya llegan muestras
            Clock.schedule_once(self._post_check, 2.0)
            print("[IMU] start() -> True")
            return True

        except Exception as e:
            print(f"[IMU] start() error: {e}")
            self.enabled = False
            IMU_AVAILABLE = False
            return False

    def _post_check(self, *_):
        print(f"[IMU] post_check acc={self.last_acc is not None} gyro={self.last_gyro is not None}")

    def stop(self):
        if platform == "android" and self.mgr and self.listener:
            try:
                self.mgr.unregisterListener(self.listener)
                print("[IMU] Listeners desregistrados")
            except Exception as e:
                print(f"[IMU] stop() error: {e}")
        self.enabled = False

    def latest(self):
        return {
            "acc": self.last_acc,   # (t_ns, [ax, ay, az]) o None
            "gyro": self.last_gyro  # (t_ns, [gx, gy, gz]) o None
        }

    def get_status(self) -> str:
        if not self.enabled:
            return "IMU: no disponible"
        acc_ok = self.last_acc is not None
        gyro_ok = self.last_gyro is not None
        if not (acc_ok or gyro_ok):
            return "IMU: iniciado, sin muestras aún"
        chk = lambda b: "✓" if b else "×"
        return f"IMU: OK (acc={chk(acc_ok)}, gyro={chk(gyro_ok)})"
