"""
gestionar_usuarios.py — Utilidad para administrar credenciales de la app.
Uso: python gestionar_usuarios.py
"""
import json
import hashlib
import os

USERS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "users.json")


def _hash(pw: str) -> str:
    return hashlib.sha256(pw.encode("utf-8")).hexdigest()


def cargar() -> dict:
    if not os.path.exists(USERS_FILE):
        return {}
    with open(USERS_FILE, encoding="utf-8") as f:
        return json.load(f)


def guardar(users: dict) -> None:
    with open(USERS_FILE, "w", encoding="utf-8") as f:
        json.dump(users, f, indent=2, ensure_ascii=False)


def listar(users: dict) -> None:
    print("\n── Usuarios registrados ──────────────────")
    if not users:
        print("  (sin usuarios)")
    for i, u in enumerate(users, 1):
        print(f"  {i}. {u}")
    print("──────────────────────────────────────────\n")


def menu() -> None:
    while True:
        users = cargar()
        print("\n╔══════════════════════════════════════╗")
        print("║   Gestión de usuarios — App Segmentación ║")
        print("╠══════════════════════════════════════╣")
        print("║  1. Ver usuarios                     ║")
        print("║  2. Cambiar contraseña               ║")
        print("║  3. Agregar usuario                  ║")
        print("║  4. Eliminar usuario                 ║")
        print("║  5. Salir                            ║")
        print("╚══════════════════════════════════════╝")
        op = input("Opción: ").strip()

        if op == "1":
            listar(users)

        elif op == "2":
            listar(users)
            usuario = input("Usuario a modificar: ").strip()
            if usuario not in users:
                print(f"  ✗ Usuario '{usuario}' no existe.")
                continue
            pw1 = input("  Nueva contraseña : ").strip()
            pw2 = input("  Confirmar        : ").strip()
            if pw1 != pw2:
                print("  ✗ Las contraseñas no coinciden.")
                continue
            if len(pw1) < 6:
                print("  ✗ La contraseña debe tener al menos 6 caracteres.")
                continue
            users[usuario] = _hash(pw1)
            guardar(users)
            print(f"  ✓ Contraseña de '{usuario}' actualizada.")

        elif op == "3":
            nuevo = input("Nombre del nuevo usuario: ").strip()
            if not nuevo:
                print("  ✗ El nombre no puede estar vacío.")
                continue
            if nuevo in users:
                print(f"  ✗ El usuario '{nuevo}' ya existe.")
                continue
            pw1 = input("  Contraseña : ").strip()
            pw2 = input("  Confirmar  : ").strip()
            if pw1 != pw2:
                print("  ✗ Las contraseñas no coinciden.")
                continue
            if len(pw1) < 6:
                print("  ✗ La contraseña debe tener al menos 6 caracteres.")
                continue
            users[nuevo] = _hash(pw1)
            guardar(users)
            print(f"  ✓ Usuario '{nuevo}' creado.")

        elif op == "4":
            listar(users)
            usuario = input("Usuario a eliminar: ").strip()
            if usuario not in users:
                print(f"  ✗ Usuario '{usuario}' no existe.")
                continue
            conf = input(f"  ¿Eliminar '{usuario}'? (s/n): ").strip().lower()
            if conf == "s":
                del users[usuario]
                guardar(users)
                print(f"  ✓ Usuario '{usuario}' eliminado.")

        elif op == "5":
            print("  Saliendo...\n")
            break
        else:
            print("  Opción no válida.")


if __name__ == "__main__":
    menu()
