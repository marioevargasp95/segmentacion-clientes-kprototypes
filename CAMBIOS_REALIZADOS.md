# Cambios realizados al proyecto — Tablero corporativo


## 1. Arranque de la app y inicio automático

- La app estaba caída (ningún proceso Python corriendo).
- Se inicia con: `python -m streamlit run app.py` desde la carpeta del proyecto.
- Se creó la tarea `Tablero[cliente]` en el Programador de Tareas de Windows para arranque automático al iniciar el servidor:
  - Usuario: SYSTEM (no requiere sesión abierta)
  - Nivel: Administrador (necesario para usar el puerto 443)
  - Trigger: Al iniciar Windows
  - Reintentos: 3 veces cada 1 minuto si falla

## 2. Instalación del certificado SSL

**Archivo fuente:** `Certificate (1).pfx` (Sectigo, wildcard `*.[cliente].com.co`)
**Contraseña del PFX:** guardada internamente, no incluir aquí.

**Comandos usados:**
```bash
# Extraer certificado con cadena completa (4 certs)
openssl pkcs12 -in "Certificate (1).pfx" -nokeys -out ssl/cert.pem -passin pass:CONTRASEÑA

# Extraer clave privada sin encriptar
openssl pkcs12 -in "Certificate (1).pfx" -nocerts -nodes -out ssl/key.pem -passin pass:CONTRASEÑA
```

**Cadena de certificados en `ssl/cert.pem`:**
1. `*.[cliente].com.co` (certificado del dominio)
2. Sectigo Public Server Authentication CA DV R36 (intermedio)
3. Sectigo Public Server Authentication Root R46 (intermedio)
4. USERTrust RSA Certification Authority (raiz)

**Configuracion en `.streamlit/config.toml`:**
```toml
[server]
address     = "0.0.0.0"
port        = 443
sslCertFile = "ssl/cert.pem"
sslKeyFile  = "ssl/key.pem"

[browser]
serverAddress = "tablero.[cliente].com.co"
serverPort    = 443
```

**Vencimiento del certificado:** 1 de septiembre de 2026.
Para renovar: repetir el proceso con el nuevo `.pfx` de Sectigo.

**Resultado SSL:**
- Protocolo: TLS 1.3
- Cifrado: AES-256-GCM
- Intercambio de claves: X25519
- Verificacion: `Verify return code: 0 (ok)`

## 3. Sesion persistente tras refresco de pagina

**Problema:** `st.session_state` se borra con cada recarga — el usuario era expulsado al refrescar.

**Librerias intentadas y descartadas:**
- `extra-streamlit-components` — incompatible con Streamlit 1.56 (CookieManager usa widgets en cache)
- `streamlit-cookies-controller` — `__cookies` es `None` en el primer render, causa `TypeError`

**Solucion final:** `st.query_params` nativo de Streamlit + sesiones en disco.

**Como funciona:**
1. Al hacer login se genera un token UUID y se guarda en `sessions.json`
2. La URL cambia a `https://tablero.[cliente].com.co/?session=TOKEN`
3. Al refrescar, la URL mantiene el `?session=TOKEN`
4. `_check_auth()` lee el query param, valida contra `sessions.json` y restaura la sesion

**Archivos modificados en `app.py`:**
- Importaciones: agregado `import uuid`
- Constantes nuevas: `INACTIVITY_TIMEOUT = 30`, `SESSIONS_FILE`
- Funciones nuevas: `_load_sessions()`, `_save_sessions()`
- Funciones modificadas: `_login_page()`, `_check_auth()`

**Cierre por inactividad:** 30 minutos. Configurable en `app.py`:
```python
INACTIVITY_TIMEOUT = 30  # cambiar a los minutos deseados
```

**Sesiones en disco:** el archivo `sessions.json` persiste entre reinicios del servidor.
Si se quiere cerrar todas las sesiones manualmente, borrar `sessions.json`.

