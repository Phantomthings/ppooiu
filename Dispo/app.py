#app.py
import math
import os
import re
from dataclasses import dataclass
from datetime import datetime, time, timedelta, timezone
from itertools import cycle
from typing import Any, Dict, Optional, List, Tuple, Set, Callable
import logging

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sqlalchemy import create_engine, text, inspect
from sqlalchemy.exc import NoSuchTableError, SQLAlchemyError


class ExclusionError(RuntimeError):
    """Raised when an exclusion operation cannot be completed."""


@dataclass
class ExclusionActionResult:
    """Represents the outcome of an exclusion related change."""

    table_name: str
    block_id: int
    exclusion_id: int
    previous_status: int
    new_status: int
    changed_by: Optional[str]
    comment: Optional[str]


_TABLE_NAME_PATTERN = re.compile(r"^[A-Za-z0-9_]+$")
from Projects import mapping_sites
from Binaire import get_equip_config, translate_ic_pc

# Config
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

MODE_EQUIPMENT = "equip"
MODE_PDC = "pdc"
MODE_LABELS = {
    MODE_EQUIPMENT: "Disponibilité équipements",
    MODE_PDC: "Disponibilité points de charge",
}
GENERIC_SCOPE_TOKENS = ("tous", "toutes", "all", "global", "ensemble", "général", "general")


def get_current_mode() -> str:
    return st.session_state.get("app_mode", MODE_EQUIPMENT)


def set_current_mode(mode: str) -> None:
    if mode not in MODE_LABELS:
        mode = MODE_EQUIPMENT
    st.session_state["app_mode"] = mode

st.set_page_config(
    layout="wide",
    page_title="Disponibilité Équipements",
    page_icon="📊",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
  .stMetric {
      background-color: #f0f2f6;
      padding: 12px;
      border-radius: 10px;
      box-shadow: 0 2px 4px rgba(0,0,0,0.1);
  }
  .stMetric label {
      font-weight: 400;
      color: #1f77b4;
  }
  div[data-testid="stExpander"] {
      background-color: #ffffff;
      border: 1px solid #e0e0e0;
      border-radius: 5px;
  }
  .success-box {
      padding: 10px;
      background-color: #d4edda;
      border-left: 4px solid #28a745;
      margin: 10px 0;
  }
  .warning-box {
      padding: 10px;
      background-color: #fff3cd;
      border-left: 4px solid #ffc107;
      margin: 10px 0;
  }
  .error-box {
      padding: 10px;
      background-color: #f8d7da;
      border-left: 4px solid #dc3545;
      margin: 10px 0;
  }

  div[data-testid="stMetricValue"] { font-size: 1.47rem !important; line-height: 1.2; }
  div[data-testid="stMetricDelta"] { font-size: 0.85rem !important; line-height: 1.1; }
  div[data-testid="stMetricLabel"] > div { font-size: 1.35rem !important; }
</style>
""", unsafe_allow_html=True)

# Config
def get_db_config() -> Dict[str, str]:
    return {
        "user": st.secrets.get("MYSQL_USER", os.getenv("MYSQL_USER", "AdminNidec")),
        "password": st.secrets.get("MYSQL_PASSWORD", os.getenv("MYSQL_PASSWORD", "u6Ehe987XBSXxa4")),
        "host": st.secrets.get("MYSQL_HOST", os.getenv("MYSQL_HOST", "141.94.31.144")),
        "port": int(st.secrets.get("MYSQL_PORT", os.getenv("MYSQL_PORT", 3306))),
        "database": st.secrets.get("MYSQL_DB", os.getenv("MYSQL_DB", "indicator"))
    }

@st.cache_resource
def get_engine():
    """Crée et retourne l'engine SQLAlchemy avec gestion d'erreurs."""
    try:
        config = get_db_config()
        engine_uri = (
            f"mysql+pymysql://{config['user']}:{config['password']}"
            f"@{config['host']}:{config['port']}/{config['database']}"
            f"?charset=utf8mb4"
        )
        engine = create_engine(
            engine_uri,
            pool_pre_ping=True,
            pool_recycle=3600,
            pool_size=5,
            max_overflow=10,
            echo=False
        )
        # Test de connexion
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        logger.info("Connexion à la base de données établie avec succès")
        return engine
    except Exception as e:
        logger.error(f"Erreur de connexion à la base de données: {e}")
        st.error(f"❌ Impossible de se connecter à la base de données: {e}")
        st.stop()

# Couche Données
class DatabaseError(Exception):
    pass

@st.cache_data(ttl=1800, show_spinner=False)
def execute_query(query: str, params: Optional[Dict] = None) -> pd.DataFrame:
    try:
        engine = get_engine()
        with engine.connect() as conn:
            if "dispo_blocs_exclusions" in query:
                try:
                    _ensure_exclusion_table(conn)
                except SQLAlchemyError as ensure_exc:
                    logger.warning(
                        "Impossible de créer la table des exclusions avant la requête: %s",
                        ensure_exc,
                    )
            df = pd.read_sql_query(text(query), conn, params=params or {})
        return df
    except SQLAlchemyError as e:
        logger.error(f"Erreur SQL: {e}")
        raise DatabaseError(f"Erreur lors de l'exécution de la requête: {str(e)}")
    except Exception as e:
        logger.error(f"Erreur inattendue: {e}")
        raise DatabaseError(f"Erreur inattendue: {str(e)}")

def execute_write(query: str, params: Optional[Dict] = None) -> bool:
    """Exécute une requête d'écriture (INSERT, UPDATE, DELETE)."""
    try:
        engine = get_engine()
        with engine.begin() as conn:
            conn.execute(text(query), params or {})
        invalidate_cache()
        return True
    except SQLAlchemyError as e:
        logger.error(f"Erreur lors de l'écriture: {e}")
        st.error(f"❌ Erreur lors de l'opération: {str(e)}")
        return False
    except Exception as e:
        logger.error(f"Erreur inattendue lors de l'écriture: {e}")
        st.error(f"❌ Erreur inattendue: {str(e)}")
        return False


def _ensure_reclassification_history_table(conn) -> None:
    """Create the history table if it does not already exist."""

    dialect = conn.dialect.name
    if dialect == "mysql":
        create_stmt = text(
            """
            CREATE TABLE IF NOT EXISTS dispo_reclassements_historique (
                id BIGINT UNSIGNED NOT NULL AUTO_INCREMENT,
                table_name VARCHAR(128) NOT NULL,
                bloc_id BIGINT UNSIGNED NOT NULL,
                ancien_est_disponible INTEGER NOT NULL,
                nouvel_est_disponible INTEGER NOT NULL,
                changed_by VARCHAR(100) DEFAULT NULL,
                commentaire TEXT DEFAULT NULL,
                changed_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (id),
                INDEX idx_dispo_reclassement_table_bloc (table_name, bloc_id, changed_at)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
            """
        )
    else:
        create_stmt = text(
            """
            CREATE TABLE IF NOT EXISTS dispo_reclassements_historique (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                table_name VARCHAR(128) NOT NULL,
                bloc_id BIGINT NOT NULL,
                ancien_est_disponible INTEGER NOT NULL,
                nouvel_est_disponible INTEGER NOT NULL,
                changed_by VARCHAR(100) DEFAULT NULL,
                commentaire TEXT DEFAULT NULL,
                changed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )

    conn.execute(create_stmt)


def _fetch_block_status(conn, table_name: str, block_id: int) -> int:
    """Return the current est_disponible value for the block."""

    status_column = _resolve_status_column(conn, table_name)
    select_stmt = text(
        f"""
        SELECT {status_column}
        FROM `{table_name}`
        WHERE id = :block_id
        """
    )
    row = conn.execute(select_stmt, {"block_id": block_id}).mappings().first()
    if row is None:
        raise ExclusionError(
            f"Bloc {block_id} introuvable dans la table {table_name}."
        )

    try:
        return int(row[status_column])
    except (TypeError, ValueError) as exc:
        raise ExclusionError(
            f"Valeur 'est_disponible' invalide pour le bloc {block_id}."
        ) from exc


_STATUS_COLUMN_CACHE: Dict[str, str] = {}
_STATUS_COLUMN_CANDIDATES: Tuple[str, ...] = ("est_disponible", "etat")


def _resolve_status_column(conn, table_name: str) -> str:
    """Return the name of the status column for the given table."""

    if table_name in _STATUS_COLUMN_CACHE:
        return _STATUS_COLUMN_CACHE[table_name]

    inspector = inspect(conn)
    try:
        columns = {col["name"] for col in inspector.get_columns(table_name)}
    except NoSuchTableError as exc:
        raise ExclusionError(
            f"Table {table_name} introuvable lors de la résolution de la colonne de statut."
        ) from exc
    except SQLAlchemyError as exc:  # pragma: no cover - safety net
        raise ExclusionError(
            f"Impossible de déterminer la colonne de statut pour {table_name}: {exc}"
        ) from exc

    for candidate in _STATUS_COLUMN_CANDIDATES:
        if candidate in columns:
            _STATUS_COLUMN_CACHE[table_name] = candidate
            return candidate

    raise ExclusionError(
        f"La table {table_name} ne contient aucune colonne de statut reconnue ({', '.join(_STATUS_COLUMN_CANDIDATES)})."
    )

def _is_valid_table_name(table_name: str) -> bool:
    return bool(_TABLE_NAME_PATTERN.match(table_name))


def _ensure_exclusion_table(conn) -> None:
    dialect = conn.dialect.name
    if dialect == "mysql":
        create_stmt = text(
            """
            CREATE TABLE IF NOT EXISTS dispo_blocs_exclusions (
                id BIGINT UNSIGNED NOT NULL AUTO_INCREMENT,
                table_name VARCHAR(128) NOT NULL,
                bloc_id BIGINT UNSIGNED NOT NULL,
                previous_status TINYINT NOT NULL,
                exclusion_comment TEXT DEFAULT NULL,
                applied_by VARCHAR(100) DEFAULT NULL,
                applied_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                released_by VARCHAR(100) DEFAULT NULL,
                release_comment TEXT DEFAULT NULL,
                released_at TIMESTAMP NULL DEFAULT NULL,
                PRIMARY KEY (id),
                UNIQUE KEY uq_block_active (table_name, bloc_id, released_at)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
            """
        )
    else:
        create_stmt = text(
            """
            CREATE TABLE IF NOT EXISTS dispo_blocs_exclusions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                table_name VARCHAR(128) NOT NULL,
                bloc_id BIGINT NOT NULL,
                previous_status INTEGER NOT NULL,
                exclusion_comment TEXT DEFAULT NULL,
                applied_by VARCHAR(100) DEFAULT NULL,
                applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                released_by VARCHAR(100) DEFAULT NULL,
                release_comment TEXT DEFAULT NULL,
                released_at TIMESTAMP DEFAULT NULL
            )
            """
        )
    conn.execute(create_stmt)


def _get_active_exclusion(conn, table_name: str, block_id: int) -> Optional[Dict[str, Any]]:
    stmt = text(
        """
        SELECT id, previous_status, exclusion_comment, applied_by, applied_at
        FROM dispo_blocs_exclusions
        WHERE table_name = :table_name
          AND bloc_id = :block_id
          AND released_at IS NULL
        FOR UPDATE
        """
    )
    row = conn.execute(stmt, {"table_name": table_name, "block_id": block_id}).mappings().first()
    return dict(row) if row else None


def apply_block_exclusion(
    table_name: str,
    block_id: int,
    *,
    user: Optional[str] = None,
    comment: Optional[str] = None,
    new_status: int = 1,
) -> ExclusionActionResult:
    if not _is_valid_table_name(table_name):
        raise ExclusionError("Nom de table invalide pour l'exclusion.")

    if new_status not in (0, 1):
        raise ExclusionError("Valeur 'est_disponible' invalide pour l'exclusion.")

    engine = get_engine()
    current_status: Optional[int] = None
    exclusion_id: Optional[int] = None

    try:
        with engine.begin() as conn:
            _ensure_exclusion_table(conn)
            _ensure_reclassification_history_table(conn)

            current_status = _fetch_block_status(conn, table_name, block_id)
            if current_status == 1 and new_status == 1:
                raise ExclusionError("Le bloc est déjà disponible, exclusion inutile.")

            existing = _get_active_exclusion(conn, table_name, block_id)
            if existing:
                raise ExclusionError("Une exclusion active existe déjà pour ce bloc.")

            update_stmt = text(
                f"""
                UPDATE `{table_name}`
                SET est_disponible = :new_status
                WHERE id = :block_id
                """
            )
            result = conn.execute(
                update_stmt,
                {"block_id": block_id, "new_status": int(new_status)},
            )
            if result.rowcount == 0:
                raise ExclusionError("Aucun bloc mis à jour lors de l'exclusion.")

            insert_stmt = text(
                """
                INSERT INTO dispo_blocs_exclusions
                    (table_name, bloc_id, previous_status, exclusion_comment, applied_by)
                VALUES
                    (:table_name, :bloc_id, :previous_status, :comment, :user)
                """
            )
            res = conn.execute(
                insert_stmt,
                {
                    "table_name": table_name,
                    "bloc_id": block_id,
                    "previous_status": current_status,
                    "comment": comment,
                    "user": user,
                },
            )
            exclusion_id = res.lastrowid

            history_stmt = text(
                """
                INSERT INTO dispo_reclassements_historique
                    (table_name, bloc_id, ancien_est_disponible,
                     nouvel_est_disponible, changed_by, commentaire)
                VALUES
                    (:table_name, :bloc_id, :old_status, :new_status,
                     :user, :comment)
                """
            )
            conn.execute(
                history_stmt,
                {
                    "table_name": table_name,
                    "bloc_id": block_id,
                    "old_status": current_status,
                    "new_status": int(new_status),
                    "user": user,
                    "comment": comment,
                },
            )
    except SQLAlchemyError as exc:
        raise ExclusionError(
            f"Erreur lors de l'exclusion du bloc {block_id} dans {table_name}: {exc}"
        ) from exc

    invalidate_cache()

    if current_status is None or exclusion_id is None:
        raise ExclusionError("Échec de la création de l'exclusion.")

    return ExclusionActionResult(
        table_name=table_name,
        block_id=block_id,
        exclusion_id=int(exclusion_id),
        previous_status=current_status,
        new_status=int(new_status),
        changed_by=user,
        comment=comment,
    )


def release_block_exclusion(
    table_name: str,
    block_id: int,
    *,
    user: Optional[str] = None,
    comment: Optional[str] = None,
) -> ExclusionActionResult:
    if not _is_valid_table_name(table_name):
        raise ExclusionError("Nom de table invalide pour l'exclusion.")

    engine = get_engine()
    active: Optional[Dict[str, Any]] = None
    current_status: Optional[int] = None

    try:
        with engine.begin() as conn:
            _ensure_exclusion_table(conn)
            _ensure_reclassification_history_table(conn)

            active = _get_active_exclusion(conn, table_name, block_id)
            if not active:
                raise ExclusionError("Aucune exclusion active à lever pour ce bloc.")

            current_status = _fetch_block_status(conn, table_name, block_id)

            restore_stmt = text(
                f"""
                UPDATE `{table_name}`
                SET est_disponible = :previous_status
                WHERE id = :block_id
                """
            )
            conn.execute(
                restore_stmt,
                {
                    "previous_status": int(active["previous_status"]),
                    "block_id": block_id,
                },
            )

            update_stmt = text(
                """
                UPDATE dispo_blocs_exclusions
                SET released_at = CURRENT_TIMESTAMP,
                    released_by = :user,
                    release_comment = :comment
                WHERE id = :exclusion_id
                """
            )
            conn.execute(
                update_stmt,
                {
                    "exclusion_id": active["id"],
                    "user": user,
                    "comment": comment,
                },
            )

            history_stmt = text(
                """
                INSERT INTO dispo_reclassements_historique
                    (table_name, bloc_id, ancien_est_disponible,
                     nouvel_est_disponible, changed_by, commentaire)
                VALUES
                    (:table_name, :bloc_id, :old_status, :new_status,
                     :user, :comment)
                """
            )
            conn.execute(
                history_stmt,
                {
                    "table_name": table_name,
                    "bloc_id": block_id,
                    "old_status": current_status,
                    "new_status": int(active["previous_status"]),
                    "user": user,
                    "comment": comment,
                },
            )
    except SQLAlchemyError as exc:
        raise ExclusionError(
            f"Erreur lors de la suppression de l'exclusion du bloc {block_id} dans {table_name}: {exc}"
        ) from exc

    invalidate_cache()

    if active is None or current_status is None:
        raise ExclusionError("Impossible de finaliser la suppression de l'exclusion.")

    return ExclusionActionResult(
        table_name=table_name,
        block_id=block_id,
        exclusion_id=int(active["id"]),
        previous_status=current_status,
        new_status=int(active["previous_status"]),
        changed_by=user,
        comment=comment,
    )


def delete_annotation(annotation_id: int) -> bool:
    """Supprime définitivement une annotation identifiée par son ID."""
    query = "DELETE FROM dispo_annotations WHERE id = :id"
    params = {"id": annotation_id}
    return execute_write(query, params)


def render_inline_delete_table(
    df: pd.DataFrame,
    *,
    column_settings: List[Tuple[str, str, float]],
    key_prefix: str,
    delete_handler: Callable[[int], bool],
    success_message: str,
    error_message: str,
) -> None:
    """Affiche un tableau avec une action de suppression en ligne pour chaque ligne."""

    if df.empty:
        st.info("ℹ️ Aucun enregistrement à afficher.")
        return

    widths = [max(width, 0.5) for _, _, width in column_settings]
    action_width = 0.7

    header_cols = st.columns(widths + [action_width])
    for col, (_, label, _) in enumerate(column_settings):
        header_cols[col].markdown(f"**{label}**")
    header_cols[-1].markdown("**Actions**")

    for idx, row in df.iterrows():
        cols = st.columns(widths + [action_width])
        row_dict = row.to_dict()

        for col, (col_name, _, _) in enumerate(column_settings):
            value = row_dict.get(col_name, "")
            display_value = "—" if pd.isna(value) or value == "" else value
            cols[col].write(display_value)

        delete_key = f"{key_prefix}_delete_{row_dict.get('id', idx)}_{idx}"

        if cols[-1].button("🗑️", key=delete_key):
            with st.spinner("Suppression en cours..."):
                try:
                    identifier = int(row_dict.get("id"))
                except (TypeError, ValueError):
                    identifier = None

                success = False
                if identifier is not None:
                    try:
                        success = bool(delete_handler(identifier))
                    except Exception as exc:  # pragma: no cover - sécurité UI
                        logger.exception("Erreur lors de la suppression: %s", exc)

                message_context = {k: row_dict.get(k) for k, _, _ in column_settings}
                message_context.update(row_dict)

                if success:
                    st.success(success_message.format(**message_context))
                    invalidate_cache()
                    st.rerun()
                else:
                    st.error(error_message.format(**message_context))


def invalidate_cache():
    """Invalide le cache de données."""
    st.cache_data.clear()
    st.session_state["last_cache_clear"] = datetime.utcnow().isoformat()
    logger.info("Cache invalidé")
@st.cache_data(ttl=1800, show_spinner=False)
def _list_ac_tables() -> pd.DataFrame:
    """
    Retourne un DF avec colonnes: site_code, table_name
    pour toutes les tables dispo_blocs_ac_<site> du schéma.
    """
    q = """
    SELECT TABLE_NAME AS table_name
    FROM information_schema.tables
    WHERE TABLE_SCHEMA = :db
      AND TABLE_NAME REGEXP '^dispo_blocs_ac_[0-9]+(_[0-9]+)?$'
    ORDER BY TABLE_NAME
    """
    df = execute_query(q, {"db": get_db_config()["database"]})
    if df.empty:
        return pd.DataFrame(columns=["site_code", "table_name"])

    df.columns = [c.lower() for c in df.columns]
    if "table_name" not in df.columns:
        return pd.DataFrame(columns=["site_code", "table_name"])

    def _parse(tbl: str) -> pd.Series:
        t = str(tbl)
        if t.startswith("dispo_blocs_ac_"):
            return pd.Series([t[len("dispo_blocs_ac_"):], t])
        return pd.Series([None, t])

    out = df["table_name"].apply(_parse)
    out.columns = ["site_code", "table_name"]
    return out.dropna(subset=["site_code"]).reset_index(drop=True)


@st.cache_data(ttl=1800, show_spinner=False)
def _list_pdc_tables() -> pd.DataFrame:
    q = """
    SELECT TABLE_NAME AS table_name
    FROM information_schema.tables
    WHERE TABLE_SCHEMA = :db
      AND TABLE_NAME REGEXP '^dispo_pdc_n[0-9]+_[0-9]+(_[0-9]+)?$'
    ORDER BY TABLE_NAME
    """
    df = execute_query(q, {"db": get_db_config()["database"]})
    if df.empty:
        return pd.DataFrame(columns=["site_code", "pdc_id", "table_name"])

    df.columns = [c.lower() for c in df.columns]
    if "table_name" not in df.columns:
        return pd.DataFrame(columns=["site_code", "pdc_id", "table_name"])

    def _parse(tbl: str) -> pd.Series:
        t = str(tbl)
        prefix = "dispo_pdc_"
        if not t.startswith(prefix):
            return pd.Series([None, None, t])
        payload = t[len(prefix):]
        parts = payload.split("_", 1)
        if len(parts) != 2:
            return pd.Series([None, None, t])
        num = parts[0].lstrip("nN")
        pdc_id = f"PDC{num}" if num else None
        return pd.Series([parts[1], pdc_id, t])

    out = df["table_name"].apply(_parse)
    out.columns = ["site_code", "pdc_id", "table_name"]
    return out.dropna(subset=["site_code", "pdc_id"]).reset_index(drop=True)

def _sanitize_scope_options(options: List[str]) -> List[str]:
    """Supprime les entrées génériques (tous/global) d'une liste."""
    cleaned: List[str] = []
    for value in options:
        if value is None:
            continue
        text = str(value).strip()
        if not text:
            continue
        lowered = text.lower()
        if any(token in lowered for token in ("tous", "toutes", "all", "global", "ensemble")):
            continue
        cleaned.append(text)
    return cleaned


def get_sites(mode: str = MODE_EQUIPMENT) -> List[str]:
    """Récupère la liste des sites en fonction du mode sélectionné."""
    if mode == MODE_PDC:
        try:
            pdc = _list_pdc_tables()
        except DatabaseError:
            pdc = pd.DataFrame(columns=["site_code"])
        if pdc.empty:
            return []
        return sorted(_sanitize_scope_options(pdc["site_code"].unique().tolist()))

    try:
        ac = _list_ac_tables()
    except DatabaseError:
        ac = pd.DataFrame(columns=["site_code"])
    try:
        bt = _list_batt_tables()
    except DatabaseError:
        bt = pd.DataFrame(columns=["site_code", "kind", "table_name"])

    ac_sites = set(ac["site_code"].tolist()) if not ac.empty else set()
    bt_sites = set(bt["site_code"].tolist()) if not bt.empty else set()
    return sorted(_sanitize_scope_options(list(ac_sites.union(bt_sites))))


def get_equipments(mode: str = MODE_EQUIPMENT, site: Optional[str] = None) -> List[str]:
    if mode == MODE_PDC:
        pdc_tbls = _list_pdc_tables()
        if pdc_tbls.empty:
            return []
        if site:
            subset = pdc_tbls[pdc_tbls["site_code"] == site]
        else:
            subset = pdc_tbls
        return sorted(_sanitize_scope_options(subset["pdc_id"].unique().tolist()))

    equips = set()
    ac_tbls = _list_ac_tables()
    bt_tbls = _list_batt_tables()

    if site:
        if not ac_tbls.empty and (ac_tbls["site_code"] == site).any():
            equips.add("AC")
        if not bt_tbls.empty and ((bt_tbls["site_code"] == site) & (bt_tbls["kind"] == "batt")).any():
            equips.add("DC1")
        if not bt_tbls.empty and ((bt_tbls["site_code"] == site) & (bt_tbls["kind"] == "batt2")).any():
            equips.add("DC2")
    else:
        if not ac_tbls.empty:
            equips.add("AC")
        if not bt_tbls.empty and (bt_tbls["kind"] == "batt").any():
            equips.add("DC1")
        if not bt_tbls.empty and (bt_tbls["kind"] == "batt2").any():
            equips.add("DC2")

    return sorted(_sanitize_scope_options(list(equips)))


def _load_blocks_equipment(site: str, equip: str, start_dt: datetime, end_dt: datetime) -> pd.DataFrame:
    params = {"site": site, "equip": equip, "start": start_dt, "end": end_dt}
    try:
        q_view = """
            SELECT *
            FROM dispo_blocs_with_exclusion_flag
            WHERE site = :site
              AND equipement_id = :equip
              AND date_debut < :end
              AND date_fin   > :start
            ORDER BY date_debut
        """
        df = execute_query(q_view, params)
        if not df.empty and {"bloc_id", "source_table"}.issubset(df.columns):
            return _normalize_blocks_df(df)
    except DatabaseError:
        pass

    batt_union = _batt_union_sql_for_site(site)
    ac_union = _ac_union_sql_for_site(site)
    q = f"""
    WITH ac AS (
        {ac_union}
    ),
    batt AS (
        {batt_union}
    ),
    base AS (
        SELECT
        bloc_id, source_table,
        site, equipement_id, type_equipement, date_debut, date_fin,
        est_disponible, cause, raw_point_count, processed_at, batch_id, hash_signature
        FROM ac
        UNION ALL
        SELECT
        bloc_id, source_table,
        site, equipement_id, type_equipement, date_debut, date_fin,
        est_disponible, cause, raw_point_count, processed_at, batch_id, hash_signature
        FROM batt
    )
    SELECT
    b.bloc_id, b.source_table,
    b.site, b.equipement_id, b.type_equipement, b.date_debut, b.date_fin,
    b.est_disponible, b.cause, b.raw_point_count, b.processed_at, b.batch_id, b.hash_signature,
    TIMESTAMPDIFF(MINUTE, GREATEST(b.date_debut, :start), LEAST(b.date_fin, :end)) AS duration_minutes,
    COALESCE(e.previous_status, b.est_disponible) AS previous_status,
    CASE WHEN e.id IS NOT NULL THEN 1 ELSE 0 END AS is_excluded,
    e.id AS exclusion_id,
    e.applied_by AS exclusion_applied_by,
    e.applied_at AS exclusion_applied_at,
    e.exclusion_comment AS exclusion_comment
    FROM base b
    LEFT JOIN dispo_blocs_exclusions e
      ON e.table_name = b.source_table
     AND e.bloc_id = b.bloc_id
     AND e.released_at IS NULL
    WHERE b.equipement_id = :equip
    AND b.date_debut < :end
    AND b.date_fin   > :start
    ORDER BY b.date_debut
    """

    df = execute_query(q, params)
    return _normalize_blocks_df(df)


def _load_blocks_pdc(site: str, equip: str, start_dt: datetime, end_dt: datetime) -> pd.DataFrame:
    params = {"site": site, "equip": equip, "start": start_dt, "end": end_dt}
    union_sql = _pdc_union_sql_for_site(site)
    q = f"""
    WITH pdc AS (
        {union_sql}
    )
    SELECT
      p.bloc_id,
      p.source_table,
      p.site,
      p.equipement_id,
      p.type_equipement,
      p.date_debut,
      p.date_fin,
      p.est_disponible,
      p.cause,
      p.raw_point_count,
      p.processed_at,
      p.batch_id,
      p.hash_signature,
      TIMESTAMPDIFF(MINUTE, GREATEST(p.date_debut, :start), LEAST(p.date_fin, :end)) AS duration_minutes,
      COALESCE(e.previous_status, p.est_disponible) AS previous_status,
      CASE WHEN e.id IS NOT NULL THEN 1 ELSE 0 END AS is_excluded,
      e.id AS exclusion_id,
      e.applied_by AS exclusion_applied_by,
      e.applied_at AS exclusion_applied_at,
      e.exclusion_comment AS exclusion_comment
    FROM pdc p
    LEFT JOIN dispo_blocs_exclusions e
      ON e.table_name = p.source_table
     AND e.bloc_id = p.bloc_id
     AND e.released_at IS NULL
    WHERE p.equipement_id = :equip
      AND p.date_debut < :end
      AND p.date_fin   > :start
    ORDER BY p.date_debut
    """

    df = execute_query(q, params)
    return _normalize_blocks_df(df)


def load_blocks(site: str, equip: str, start_dt: datetime, end_dt: datetime, mode: Optional[str] = None) -> pd.DataFrame:
    active_mode = mode or get_current_mode()
    if active_mode == MODE_PDC:
        return _load_blocks_pdc(site, equip, start_dt, end_dt)
    return _load_blocks_equipment(site, equip, start_dt, end_dt)

def _load_filtered_blocks_equipment(start_dt: datetime, end_dt: datetime, site: Optional[str] = None, equip: Optional[str] = None) -> pd.DataFrame:
    params = {"start": start_dt, "end": end_dt}
    try:
        filters = ["date_debut < :end", "date_fin > :start"]
        if site:
            filters.append("site = :site"); params["site"] = site
        if equip:
            filters.append("equipement_id = :equip"); params["equip"] = equip

        q_view = f"""
            SELECT * FROM dispo_blocs_with_exclusion_flag
            WHERE {' AND '.join(filters)}
            ORDER BY date_debut
        """
        df = execute_query(q_view, params)
        if not df.empty:
            normalized = _normalize_blocks_df(df)
            return _clip_block_durations(normalized, start_dt, end_dt)
    except DatabaseError:
        pass

    if site:
        ac_union = _ac_union_sql_for_site(site)
        batt_union = _batt_union_sql_for_site(site)
    else:
        ac_union = _ac_union_sql_all_sites()
        batt_union = _batt_union_sql_all_sites()

    equip_filter = "AND b.equipement_id = :equip" if equip else ""

    q = f"""
    WITH ac AS (
        {ac_union}
    ),
    batt AS (
        {batt_union}
    ),
    base AS (
        SELECT
        bloc_id, source_table,
        site, equipement_id, type_equipement, date_debut, date_fin,
        est_disponible, cause, raw_point_count, processed_at, batch_id, hash_signature
        FROM ac
        UNION ALL
        SELECT
        bloc_id, source_table,
        site, equipement_id, type_equipement, date_debut, date_fin,
        est_disponible, cause, raw_point_count, processed_at, batch_id, hash_signature
        FROM batt
    )
    SELECT
    b.bloc_id, b.source_table,
    b.site, b.equipement_id, b.type_equipement, b.date_debut, b.date_fin,
    b.est_disponible, b.cause, b.raw_point_count, b.processed_at, b.batch_id, b.hash_signature,
    TIMESTAMPDIFF(MINUTE, GREATEST(b.date_debut, :start), LEAST(b.date_fin, :end)) AS duration_minutes,
    COALESCE(e.previous_status, b.est_disponible) AS previous_status,
    CASE WHEN e.id IS NOT NULL THEN 1 ELSE 0 END AS is_excluded,
    e.id AS exclusion_id,
    e.applied_by AS exclusion_applied_by,
    e.applied_at AS exclusion_applied_at,
    e.exclusion_comment AS exclusion_comment
    FROM base b
    LEFT JOIN dispo_blocs_exclusions e
      ON e.table_name = b.source_table
     AND e.bloc_id = b.bloc_id
     AND e.released_at IS NULL
    WHERE b.date_debut < :end
    AND b.date_fin   > :start
    {equip_filter}
    ORDER BY b.date_debut
    """

    df = execute_query(q, params)
    normalized = _normalize_blocks_df(df)
    return _clip_block_durations(normalized, start_dt, end_dt)


def _load_filtered_blocks_pdc(start_dt: datetime, end_dt: datetime, site: Optional[str] = None, equip: Optional[str] = None) -> pd.DataFrame:
    params = {"start": start_dt, "end": end_dt}
    if site:
        union_sql = _pdc_union_sql_for_site(site)
    else:
        union_sql = _pdc_union_sql_all_sites()
    if site:
        params["site"] = site
    if equip:
        params["equip"] = equip

    site_filter = "AND p.site = :site" if site else ""
    equip_filter = "AND p.equipement_id = :equip" if equip else ""

    q = f"""
    WITH pdc AS (
        {union_sql}
    )
    SELECT
      p.bloc_id,
      p.source_table,
      p.site,
      p.equipement_id,
      p.type_equipement,
      p.date_debut,
      p.date_fin,
      p.est_disponible,
      p.cause,
      p.raw_point_count,
      p.processed_at,
      p.batch_id,
      p.hash_signature,
      TIMESTAMPDIFF(MINUTE, GREATEST(p.date_debut, :start), LEAST(p.date_fin, :end)) AS duration_minutes,
      COALESCE(e.previous_status, p.est_disponible) AS previous_status,
      CASE WHEN e.id IS NOT NULL THEN 1 ELSE 0 END AS is_excluded,
      e.id AS exclusion_id,
      e.applied_by AS exclusion_applied_by,
      e.applied_at AS exclusion_applied_at,
      e.exclusion_comment AS exclusion_comment
    FROM pdc p
    LEFT JOIN dispo_blocs_exclusions e
      ON e.table_name = p.source_table
     AND e.bloc_id = p.bloc_id
     AND e.released_at IS NULL
    WHERE p.date_debut < :end
      AND p.date_fin > :start
      {site_filter}
      {equip_filter}
    ORDER BY p.date_debut
    """

    df = execute_query(q, params)
    normalized = _normalize_blocks_df(df)
    return _clip_block_durations(normalized, start_dt, end_dt)


def load_filtered_blocks(start_dt: datetime, end_dt: datetime, site: Optional[str] = None, equip: Optional[str] = None, mode: Optional[str] = None) -> pd.DataFrame:
    active_mode = mode or get_current_mode()
    if active_mode == MODE_PDC:
        return _load_filtered_blocks_pdc(start_dt, end_dt, site, equip)
    return _load_filtered_blocks_equipment(start_dt, end_dt, site, equip)


def _bulk_exclude_missing_blocks(
    *,
    site: str,
    equip: str,
    start_dt: datetime,
    end_dt: datetime,
    new_status: int,
    comment: str,
    user: Optional[str],
) -> Tuple[int, int, List[str]]:
    """Applique une exclusion automatique sur les blocs manquants d'un équipement.

    Args:
        site: Code site concerné.
        equip: Identifiant équipement (AC, DC1, PDC1, ...).
        start_dt: Début de la fenêtre d'analyse.
        end_dt: Fin de la fenêtre d'analyse.
        new_status: Statut à appliquer (1 = disponible, 0 = indisponible).
        comment: Commentaire associé à l'exclusion.
        user: Opérateur ayant déclenché l'opération.

    Returns:
        Un tuple (nb_exclusions_créées, nb_blocs_candidats, liste_erreurs).
    """

    mode = MODE_PDC if equip.upper().startswith("PDC") else MODE_EQUIPMENT

    try:
        df_blocks = load_filtered_blocks(start_dt, end_dt, site, equip, mode=mode)
    except DatabaseError as exc:
        return 0, 0, [f"{equip}: impossible de charger les blocs ({exc})"]

    if df_blocks.empty:
        return 0, 0, []

    mask_missing = (df_blocks["est_disponible"] == -1) & (df_blocks["is_excluded"] == 0)
    missing_blocks = df_blocks.loc[mask_missing]

    if missing_blocks.empty:
        return 0, 0, []

    created = 0
    errors: List[str] = []

    for _, block in missing_blocks.iterrows():
        table_name = str(block.get("source_table") or "").strip()
        try:
            block_id = int(block.get("bloc_id", -1))
        except (TypeError, ValueError):
            block_id = -1

        if not table_name or block_id <= 0:
            errors.append(f"{equip}: bloc invalide (table='{table_name}' id={block_id}).")
            continue

        try:
            apply_block_exclusion(
                table_name=table_name,
                block_id=block_id,
                user=user,
                comment=comment,
                new_status=new_status,
            )
            created += 1
        except ExclusionError as exc:
            errors.append(f"{equip} · bloc {block_id}: {exc}")

    return created, len(missing_blocks), errors

# Gestion
def _insert_annotation(
    site: str,
    equip: str,
    start_dt: datetime,
    end_dt: datetime,
    annotation_type: str,
    comment: str,
    user: str = "ui",
) -> bool:
    """Insère une annotation sans logique additionnelle."""
    query = """
        INSERT INTO dispo_annotations
        (site, equipement_id, date_debut, date_fin, type_annotation, commentaire, actif, created_by)
        VALUES (:site, :equip, :start, :end, :type, :comment, 1, :user)
    """
    params = {
        "site": site,
        "equip": equip,
        "start": start_dt,
        "end": end_dt,
        "type": annotation_type,
        "comment": comment,
        "user": user
    }
    return execute_write(query, params)


def create_annotation(
    site: str,
    equip: str,
    start_dt: datetime,
    end_dt: datetime,
    annotation_type: str,
    comment: str,
    user: str = "ui",
    cascade: bool = True,
) -> bool:
    """Crée une nouvelle annotation et applique les éventuelles règles métiers."""
    success = _insert_annotation(
        site=site,
        equip=equip,
        start_dt=start_dt,
        end_dt=end_dt,
        annotation_type=annotation_type,
        comment=comment,
        user=user,
    )

    if not success:
        return False

    if (
        cascade
        and annotation_type == "exclusion"
        and equip
        and equip.upper().startswith("AC")
    ):
        for idx in range(1, 7):
            _insert_annotation(
                site=site,
                equip=f"PDC{idx}",
                start_dt=start_dt,
                end_dt=end_dt,
                annotation_type=annotation_type,
                comment=comment,
                user=user,
            )

    return True

@st.cache_data(ttl=1800, show_spinner=False)
def _list_batt_tables() -> pd.DataFrame:
    """
    Retourne un DF avec colonnes: site_code, kind ('batt'|'batt2'), table_name
    pour toutes les tables dispo_blocs_batt_* et dispo_blocs_batt2_* du schéma.
    """
    q = """
    SELECT TABLE_NAME AS table_name
    FROM information_schema.tables
    WHERE TABLE_SCHEMA = :db
      AND (
            TABLE_NAME REGEXP '^dispo_blocs_batt_[0-9]+(_[0-9]+)?$'
         OR TABLE_NAME REGEXP '^dispo_blocs_batt2_[0-9]+(_[0-9]+)?$'
      )
    ORDER BY TABLE_NAME
    """
    df = execute_query(q, {"db": get_db_config()["database"]})
    if df.empty:
        return pd.DataFrame(columns=["site_code", "kind", "table_name"])

    df.columns = [c.lower() for c in df.columns]
    if "table_name" not in df.columns:
        return pd.DataFrame(columns=["site_code", "kind", "table_name"])

    def _parse(tbl: str) -> pd.Series:
        t = str(tbl)
        if t.startswith("dispo_blocs_batt2_"):
            return pd.Series([t[len("dispo_blocs_batt2_"):], "batt2", t])
        if t.startswith("dispo_blocs_batt_"):
            return pd.Series([t[len("dispo_blocs_batt_"):], "batt", t])
        return pd.Series([None, None, t])

    out = df["table_name"].apply(_parse)
    out.columns = ["site_code", "kind", "table_name"]
    return out.dropna(subset=["site_code","kind"]).reset_index(drop=True)

@st.cache_data(ttl=1800, show_spinner=False)
def _ac_union_sql_for_site(site: str) -> str:
    """
    UNION ALL des tables AC du site (colonnes explicites, sans duration_minutes).
    """
    df = _list_ac_tables()
    if df.empty:
        return """SELECT * FROM (
            SELECT CAST(NULL AS SIGNED) AS bloc_id,
                   CAST(NULL AS CHAR) AS source_table,
                   CAST(NULL AS CHAR) AS site,
                   CAST(NULL AS CHAR) AS equipement_id,
                   CAST(NULL AS CHAR) AS type_equipement,
                   CAST(NULL AS DATETIME) AS date_debut,
                   CAST(NULL AS DATETIME) AS date_fin,
                   CAST(NULL AS SIGNED) AS est_disponible,
                   CAST(NULL AS CHAR) AS cause,
                   CAST(NULL AS SIGNED) AS raw_point_count,
                   CAST(NULL AS DATETIME) AS processed_at,
                   CAST(NULL AS CHAR) AS batch_id,
                   CAST(NULL AS CHAR) AS hash_signature
        ) x WHERE 1=0"""

    m = df[df["site_code"] == site]
    if m.empty:
        return """SELECT * FROM (
            SELECT CAST(NULL AS SIGNED) AS bloc_id,
                   CAST(NULL AS CHAR) AS source_table,
                   CAST(NULL AS CHAR) AS site,
                   CAST(NULL AS CHAR) AS equipement_id,
                   CAST(NULL AS CHAR) AS type_equipement,
                   CAST(NULL AS DATETIME) AS date_debut,
                   CAST(NULL AS DATETIME) AS date_fin,
                   CAST(NULL AS SIGNED) AS est_disponible,
                   CAST(NULL AS CHAR) AS cause,
                   CAST(NULL AS SIGNED) AS raw_point_count,
                   CAST(NULL AS DATETIME) AS processed_at,
                   CAST(NULL AS CHAR) AS batch_id,
                   CAST(NULL AS CHAR) AS hash_signature
        ) x WHERE 1=0"""

    parts = []
    for _, r in m.iterrows():
        tbl = r["table_name"]
        parts.append(f"""
            SELECT
              id AS bloc_id,
              '{tbl}' AS source_table,
              site, equipement_id, type_equipement, date_debut, date_fin,
              est_disponible, cause, raw_point_count, processed_at, batch_id, hash_signature
            FROM `{tbl}`
        """)
    return " UNION ALL ".join(parts)

@st.cache_data(ttl=1800, show_spinner=False)
def _ac_union_sql_all_sites() -> str:
    """
    UNION ALL de toutes les tables AC (colonnes explicites, sans duration_minutes).
    """
    df = _list_ac_tables()
    if df.empty:
        return """SELECT * FROM (
            SELECT CAST(NULL AS SIGNED) AS bloc_id,
                   CAST(NULL AS CHAR) AS source_table,
                   CAST(NULL AS CHAR) AS site,
                   CAST(NULL AS CHAR) AS equipement_id,
                   CAST(NULL AS CHAR) AS type_equipement,
                   CAST(NULL AS DATETIME) AS date_debut,
                   CAST(NULL AS DATETIME) AS date_fin,
                   CAST(NULL AS SIGNED) AS est_disponible,
                   CAST(NULL AS CHAR) AS cause,
                   CAST(NULL AS SIGNED) AS raw_point_count,
                   CAST(NULL AS DATETIME) AS processed_at,
                   CAST(NULL AS CHAR) AS batch_id,
                   CAST(NULL AS CHAR) AS hash_signature
        ) x WHERE 1=0"""
    parts = [
        f"""SELECT
              id AS bloc_id,
              '{tbl}' AS source_table,
              site, equipement_id, type_equipement, date_debut, date_fin,
              est_disponible, cause, raw_point_count, processed_at, batch_id, hash_signature
            FROM `{tbl}`"""
        for tbl in df["table_name"].tolist()
    ]
    return " UNION ALL ".join(parts)

@st.cache_data(ttl=1800, show_spinner=False)
def _batt_union_sql_for_site(site: str) -> str:
    """
    UNION ALL des tables BATT/BATT2 du site, en listant explicitement les colonnes
    (pas de duration_minutes ici).
    """
    df = _list_batt_tables()
    if df.empty:
        return """SELECT * FROM (
            SELECT CAST(NULL AS SIGNED) AS bloc_id,
                   CAST(NULL AS CHAR) AS source_table,
                   CAST(NULL AS CHAR) AS site,
                   CAST(NULL AS CHAR) AS equipement_id,
                   CAST(NULL AS CHAR) AS type_equipement,
                   CAST(NULL AS DATETIME) AS date_debut,
                   CAST(NULL AS DATETIME) AS date_fin,
                   CAST(NULL AS SIGNED) AS est_disponible,
                   CAST(NULL AS CHAR) AS cause,
                   CAST(NULL AS SIGNED) AS raw_point_count,
                   CAST(NULL AS DATETIME) AS processed_at,
                   CAST(NULL AS CHAR) AS batch_id,
                   CAST(NULL AS CHAR) AS hash_signature
        ) x WHERE 1=0"""

    parts = []
    for _, r in df[df["site_code"] == site].iterrows():
        tbl = r["table_name"]
        parts.append(f"""
            SELECT
              id AS bloc_id,
              '{tbl}' AS source_table,
              site, equipement_id, type_equipement, date_debut, date_fin,
              est_disponible, cause, raw_point_count, processed_at, batch_id, hash_signature
            FROM `{tbl}`
        """)
    if not parts:
        return """SELECT * FROM (
            SELECT CAST(NULL AS SIGNED) AS bloc_id,
                   CAST(NULL AS CHAR) AS source_table,
                   CAST(NULL AS CHAR) AS site,
                   CAST(NULL AS CHAR) AS equipement_id,
                   CAST(NULL AS CHAR) AS type_equipement,
                   CAST(NULL AS DATETIME) AS date_debut,
                   CAST(NULL AS DATETIME) AS date_fin,
                   CAST(NULL AS SIGNED) AS est_disponible,
                   CAST(NULL AS CHAR) AS cause,
                   CAST(NULL AS SIGNED) AS raw_point_count,
                   CAST(NULL AS DATETIME) AS processed_at,
                   CAST(NULL AS CHAR) AS batch_id,
                   CAST(NULL AS CHAR) AS hash_signature
        ) x WHERE 1=0"""
    return " UNION ALL ".join(parts)

@st.cache_data(ttl=1800, show_spinner=False)
def _batt_union_sql_all_sites() -> str:
    """
    UNION ALL de toutes les tables BATT/BATT2 (pas de duration_minutes ici).
    """
    df = _list_batt_tables()
    if df.empty:
        return """SELECT * FROM (
            SELECT CAST(NULL AS SIGNED) AS bloc_id,
                   CAST(NULL AS CHAR) AS source_table,
                   CAST(NULL AS CHAR) AS site,
                   CAST(NULL AS CHAR) AS equipement_id,
                   CAST(NULL AS CHAR) AS type_equipement,
                   CAST(NULL AS DATETIME) AS date_debut,
                   CAST(NULL AS DATETIME) AS date_fin,
                   CAST(NULL AS SIGNED) AS est_disponible,
                   CAST(NULL AS CHAR) AS cause,
                   CAST(NULL AS SIGNED) AS raw_point_count,
                   CAST(NULL AS DATETIME) AS processed_at,
                   CAST(NULL AS CHAR) AS batch_id,
                   CAST(NULL AS CHAR) AS hash_signature
        ) x WHERE 1=0"""
    parts = [f"""
        SELECT
          id AS bloc_id,
          '{tbl}' AS source_table,
          site, equipement_id, type_equipement, date_debut, date_fin,
          est_disponible, cause, raw_point_count, processed_at, batch_id, hash_signature
        FROM `{tbl}`
    """ for tbl in df["table_name"].tolist()]
    return " UNION ALL ".join(parts)


@st.cache_data(ttl=1800, show_spinner=False)
def _pdc_union_sql_for_site(site: str) -> str:
    df = _list_pdc_tables()
    if df.empty:
        return """SELECT * FROM (
            SELECT CAST(NULL AS SIGNED) AS bloc_id,
                   CAST(NULL AS CHAR) AS source_table,
                   CAST(NULL AS CHAR) AS site,
                   CAST(NULL AS CHAR) AS equipement_id,
                   CAST(NULL AS CHAR) AS type_equipement,
                   CAST(NULL AS DATETIME) AS date_debut,
                   CAST(NULL AS DATETIME) AS date_fin,
                   CAST(NULL AS SIGNED) AS est_disponible,
                   CAST(NULL AS CHAR) AS cause,
                   CAST(NULL AS SIGNED) AS raw_point_count,
                   CAST(NULL AS DATETIME) AS processed_at,
                   CAST(NULL AS CHAR) AS batch_id,
                   CAST(NULL AS CHAR) AS hash_signature
        ) x WHERE 1=0"""

    subset = df[df["site_code"] == site]
    if subset.empty:
        return """SELECT * FROM (
            SELECT CAST(NULL AS CHAR) AS site,
                   CAST(NULL AS CHAR) AS equipement_id,
                   CAST(NULL AS CHAR) AS type_equipement,
                   CAST(NULL AS DATETIME) AS date_debut,
                   CAST(NULL AS DATETIME) AS date_fin,
                   CAST(NULL AS SIGNED) AS est_disponible,
                   CAST(NULL AS CHAR) AS cause,
                   CAST(NULL AS SIGNED) AS raw_point_count,
                   CAST(NULL AS DATETIME) AS processed_at,
                   CAST(NULL AS CHAR) AS batch_id,
                   CAST(NULL AS CHAR) AS hash_signature
        ) x WHERE 1=0"""

    parts = []
    for _, row in subset.iterrows():
        tbl = row["table_name"]
        parts.append(f"""
            SELECT
              id AS bloc_id,
              '{tbl}' AS source_table,
              site,
              pdc_id AS equipement_id,
              type_label AS type_equipement,
              date_debut,
              date_fin,
              est_disponible,
              cause,
              raw_point_count,
              processed_at,
              batch_id,
              hash_signature
            FROM `{tbl}`
        """)
    return " UNION ALL ".join(parts)
@st.cache_data(ttl=1800, show_spinner=False)
def _pdc_union_sql_all_sites() -> str:
    df = _list_pdc_tables()
    if df.empty:
        return """SELECT * FROM (
            SELECT CAST(NULL AS CHAR) AS site,
                   CAST(NULL AS CHAR) AS equipement_id,
                   CAST(NULL AS CHAR) AS type_equipement,
                   CAST(NULL AS DATETIME) AS date_debut,
                   CAST(NULL AS DATETIME) AS date_fin,
                   CAST(NULL AS SIGNED) AS est_disponible,
                   CAST(NULL AS CHAR) AS cause,
                   CAST(NULL AS SIGNED) AS raw_point_count,
                   CAST(NULL AS DATETIME) AS processed_at,
                   CAST(NULL AS CHAR) AS batch_id,
                   CAST(NULL AS CHAR) AS hash_signature
        ) x WHERE 1=0"""

    parts = [
        f"""
            SELECT
              id AS bloc_id,
              '{tbl}' AS source_table,
              site,
              pdc_id AS equipement_id,
              type_label AS type_equipement,
              date_debut,
              date_fin,
              est_disponible,
              cause,
              raw_point_count,
              processed_at,
              batch_id,
              hash_signature
            FROM `{tbl}`
        """
        for tbl in df["table_name"].tolist()
    ]
    return " UNION ALL ".join(parts)

def _normalize_blocks_df(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    out = df.copy()
    for col in ["date_debut", "date_fin", "processed_at", "exclusion_applied_at"]:
        if col in out.columns:
            s = pd.to_datetime(out[col], errors="coerce")
            try:
                if s.dt.tz is None:
                    s = s.dt.tz_localize("Europe/Paris", nonexistent="shift_forward", ambiguous="NaT")
                else:
                    s = s.dt.tz_convert("Europe/Paris")
            except Exception:
                pass
            out[col] = s
    for col in [
        "est_disponible",
        "raw_point_count",
        "duration_minutes",
        "is_excluded",
        "previous_status",
        "exclusion_id",
    ]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0).astype(int)
        else:
            if col == "is_excluded":
                out[col] = 0
            elif col == "previous_status":
                out[col] = 0
            elif col == "exclusion_id":
                out[col] = -1
    if "bloc_id" in out.columns:
        out["bloc_id"] = pd.to_numeric(out["bloc_id"], errors="coerce").fillna(-1).astype(int)
    elif "id" in out.columns:
        out["bloc_id"] = pd.to_numeric(out["id"], errors="coerce").fillna(-1).astype(int)
    else:
        out["bloc_id"] = -1
    if "source_table" in out.columns:
        out["source_table"] = out["source_table"].fillna("").astype(str)
    else:
        out["source_table"] = ""
    if "previous_status" in out.columns:
        mask_no_exclusion = out.get("exclusion_id", -1) < 0
        out.loc[mask_no_exclusion, "previous_status"] = out.loc[mask_no_exclusion, "est_disponible"]
    else:
        out["previous_status"] = out.get("est_disponible", 0)
    for text_col in ["exclusion_comment", "exclusion_applied_by"]:
        if text_col in out.columns:
            out[text_col] = out[text_col].fillna("").astype(str)
    return out.sort_values("date_debut").reset_index(drop=True)


def _clip_block_durations(
    df: pd.DataFrame,
    start_dt: datetime,
    end_dt: datetime,
) -> pd.DataFrame:
    """Ajuste les durées pour ne conserver que l'intervalle analysé."""

    if df is None or df.empty:
        return df if df is not None else pd.DataFrame()

    start_ts = _ensure_paris_timestamp(start_dt)
    end_ts = _ensure_paris_timestamp(end_dt)

    if start_ts is None or end_ts is None:
        return df

    clipped = df.copy()
    clip_start = clipped["date_debut"].clip(lower=start_ts)
    clip_end = clipped["date_fin"].clip(upper=end_ts)

    duration = (
        (clip_end - clip_start).dt.total_seconds() / 60
    ).fillna(0)
    duration = duration.clip(lower=0)

    clipped["duration_minutes"] = duration.round().astype(int)

    return clipped


def _aggregate_monthly_availability(
    df: pd.DataFrame,
    start_dt: datetime,
    end_dt: datetime,
) -> pd.DataFrame:
    """Agrège les blocs de disponibilité par mois pour une période donnée."""
    if df is None or df.empty:
        return pd.DataFrame(columns=["month", "pct_brut", "pct_excl", "total_minutes"])

    df = df.copy()

    start_p = pd.Timestamp(start_dt)
    end_p = pd.Timestamp(end_dt)

    if start_p.tz is None:
        start_p = start_p.tz_localize("Europe/Paris", nonexistent="shift_forward", ambiguous="NaT")
    else:
        start_p = start_p.tz_convert("Europe/Paris")

    if end_p.tz is None:
        end_p = end_p.tz_localize("Europe/Paris", nonexistent="shift_forward", ambiguous="NaT")
    else:
        end_p = end_p.tz_convert("Europe/Paris")

    df["clip_start"] = df["date_debut"].clip(lower=start_p)
    df["clip_end"] = df["date_fin"].clip(upper=end_p)

    df = df.loc[df["clip_start"].notna() & df["clip_end"].notna()].copy()
    if df.empty:
        return pd.DataFrame(columns=["month", "pct_brut", "pct_excl", "total_minutes"])

    df["duration_minutes_window"] = (
        (df["clip_end"] - df["clip_start"]).dt.total_seconds() / 60
    ).clip(lower=0).fillna(0).astype(int)

    df["month"] = df["clip_start"].dt.to_period("M").dt.to_timestamp()

    rows: List[Dict[str, float]] = []
    for month, group in df.groupby("month"):
        total = int(group["duration_minutes_window"].sum())
        if total <= 0:
            rows.append({"month": month, "pct_brut": 0.0, "pct_excl": 0.0, "total_minutes": 0})
            continue

        # CORRECTION: Inverser la logique
        current_status = group["est_disponible"]
        
        # Disponibilité BRUTE = statut AVANT exclusions (previous_status pour les exclus)
        if "is_excluded" in group.columns and "previous_status" in group.columns:
            brut_status = current_status.where(group["is_excluded"] == 0, group["previous_status"])
        else:
            brut_status = current_status
        
        avail_brut = int(group.loc[brut_status == 1, "duration_minutes_window"].sum())
        
        # Disponibilité AVEC EXCLUSIONS = statut ACTUEL (est_disponible)
        avail_excl = int(group.loc[current_status == 1, "duration_minutes_window"].sum())

        rows.append(
            {
                "month": month,
                "pct_brut": avail_brut / total * 100.0,
                "pct_excl": avail_excl / total * 100.0,
                "total_minutes": total,
            }
        )

    return pd.DataFrame(rows).sort_values("month").reset_index(drop=True)

def toggle_annotation(annotation_id: int, active: bool) -> bool:
    """Active ou désactive une annotation."""
    query = "UPDATE dispo_annotations SET actif = :active WHERE id = :id"
    params = {"active": int(active), "id": annotation_id}
    return execute_write(query, params)

def update_annotation_comment(annotation_id: int, comment: str) -> bool:
    """Met à jour le commentaire d'une annotation."""
    query = "UPDATE dispo_annotations SET commentaire = :comment WHERE id = :id"
    params = {"comment": comment, "id": annotation_id}
    return execute_write(query, params)

def get_annotations(annotation_type: Optional[str] = None, limit: int = 200) -> pd.DataFrame:
    """Récupère les annotations."""
    query = """
        SELECT id, site, equipement_id, date_debut, date_fin,
               type_annotation, commentaire, actif, created_by, created_at
        FROM dispo_annotations
    """
    params = {}
    
    if annotation_type:
        query += " WHERE type_annotation = :type"
        params["type"] = annotation_type
    
    query += " ORDER BY created_at DESC LIMIT :limit"
    params["limit"] = limit
    
    try:
        return execute_query(query, params)
    except DatabaseError as e:
        st.error(f"Erreur lors du chargement des annotations: {e}")
        return pd.DataFrame()


@st.cache_data(ttl=300, show_spinner=False)
def get_block_exclusions(active_only: bool = True, limit: int = 200) -> pd.DataFrame:
    """Récupère les exclusions enregistrées directement sur les blocs."""

    query = """
        SELECT id, table_name, bloc_id, previous_status,
               exclusion_comment, applied_by, applied_at,
               released_by, released_at, release_comment
        FROM dispo_blocs_exclusions
    """
    params = {"limit": limit}
    if active_only:
        query += " WHERE released_at IS NULL"
    query += " ORDER BY applied_at DESC LIMIT :limit"

    try:
        engine = get_engine()
        with engine.begin() as conn:
            _ensure_exclusion_table(conn)
        return execute_query(query, params)
    except DatabaseError as exc:
        st.error(f"Erreur lors du chargement des exclusions: {exc}")
        return pd.DataFrame()

# Calculs mois
def calculate_availability(
    df: Optional[pd.DataFrame],
    include_exclusions: bool = False
) -> Dict[str, float]:
    """
    Calcule la disponibilité.
    
    Args:
        include_exclusions: 
            - False = Disponibilité brute (statut AVANT exclusions = previous_status)
            - True = Disponibilité avec exclusions (statut ACTUEL = est_disponible)
    """
    if df is None or df.empty:
        return {
            "total_minutes": 0,
            "effective_minutes": 0,
            "available_minutes": 0,
            "unavailable_minutes": 0,
            "missing_minutes": 0,
            "pct_available": 0.0,
            "pct_unavailable": 0.0,
        }
    
    total = int(df["duration_minutes"].sum())
    
    # INVERSION DE LA LOGIQUE
    if include_exclusions:
        # Disponibilité avec exclusions = statut ACTUEL (modifié par les exclusions)
        status_series = df["est_disponible"].copy()
    else:
        # Disponibilité brute = statut AVANT exclusions
        status_series = df["est_disponible"].copy()
        
        # Restaurer le statut précédent pour les blocs exclus
        if "is_excluded" in df.columns and "previous_status" in df.columns:
            # Pour les blocs exclus, prendre le previous_status
            mask_excluded = df["is_excluded"] == 1
            status_series = status_series.where(~mask_excluded, df["previous_status"])
    
    missing_minutes = int(df.loc[status_series == -1, "duration_minutes"].sum())
    available = int(df.loc[status_series == 1, "duration_minutes"].sum())
    unavailable = int(df.loc[status_series == 0, "duration_minutes"].sum())
    effective_total = available + unavailable
    
    pct_available = (available / effective_total * 100) if effective_total > 0 else 0.0
    pct_unavailable = (unavailable / effective_total * 100) if effective_total > 0 else 0.0
    
    return {
        "total_minutes": total,
        "effective_minutes": effective_total,
        "available_minutes": available,
        "unavailable_minutes": unavailable,
        "missing_minutes": missing_minutes,
        "pct_available": pct_available,
        "pct_unavailable": pct_unavailable,
    }
def _station_equipment_modes() -> List[Tuple[str, str]]:
    equipments = [("AC", MODE_EQUIPMENT), ("DC1", MODE_EQUIPMENT), ("DC2", MODE_EQUIPMENT)]
    equipments.extend([(f"PDC{i}", MODE_PDC) for i in range(1, 7)])
    return equipments


def _ensure_paris_timestamp(value: Any) -> Optional[pd.Timestamp]:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    try:
        ts = pd.Timestamp(value)
    except Exception:
        return None

    try:
        if ts.tzinfo is None:
            ts = ts.tz_localize("Europe/Paris", nonexistent="shift_forward", ambiguous="infer")
        else:
            ts = ts.tz_convert("Europe/Paris")
    except Exception:
        try:
            ts = ts.tz_localize("Europe/Paris", nonexistent="shift_forward", ambiguous="NaT")
        except Exception:
            return None

    if pd.isna(ts):
        return None
    return ts


def _build_station_timeline_df(timelines: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    records: List[Dict[str, Any]] = []
    for equip, df in timelines.items():
        if df is None or df.empty:
            continue
        for _, row in df.iterrows():
            start_ts = _ensure_paris_timestamp(row.get("date_debut"))
            end_ts = _ensure_paris_timestamp(row.get("date_fin"))
            if start_ts is None or end_ts is None or end_ts <= start_ts:
                continue
            records.append(
                {
                    "Equipement": equip,
                    "start": start_ts,
                    "end": end_ts,
                    "est_disponible": int(row.get("est_disponible", 0)),
                    "is_excluded": int(row.get("is_excluded", 0)),
                    "cause": row.get("cause"),
                    "duration_minutes": int(row.get("duration_minutes", 0)),
                }
            )

    timeline_df = pd.DataFrame.from_records(records)
    if timeline_df.empty:
        return timeline_df

    state_map = {
        1: "✅ Disponible",
        0: "❌ Indisponible",
        -1: "⚠️ Donnée manquante",
    }
    timeline_df["state"] = timeline_df["est_disponible"].map(state_map).fillna("❓ Inconnu")
    timeline_df["label"] = timeline_df["state"]
    mask_excl = timeline_df["is_excluded"] == 1
    timeline_df.loc[mask_excl, "label"] = timeline_df.loc[mask_excl, "state"] + " (Exclu)"
    return timeline_df.sort_values(["Equipement", "start"]).reset_index(drop=True)


def _new_condition_tracker(label: str) -> Dict[str, Any]:
    return {
        "label": label,
        "duration": 0.0,
        "occurrences": 0,
        "intervals": [],
        "active": False,
        "current_start": None,
        "denom": 0.0,
    }


def _update_condition_tracker(
    tracker: Dict[str, Any],
    is_active: bool,
    has_data: bool,
    seg_start: pd.Timestamp,
    seg_end: pd.Timestamp,
    duration: float,
) -> None:
    if has_data:
        tracker["denom"] += duration
    if not has_data:
        if tracker["active"]:
            tracker["intervals"].append((tracker["current_start"], seg_start))
            tracker["occurrences"] += 1
            tracker["active"] = False
            tracker["current_start"] = None
        return

    if is_active:
        tracker["duration"] += duration
        if not tracker["active"]:
            tracker["active"] = True
            tracker["current_start"] = seg_start
    else:
        if tracker["active"]:
            tracker["intervals"].append((tracker["current_start"], seg_start))
            tracker["occurrences"] += 1
            tracker["active"] = False
            tracker["current_start"] = None


def _finalize_condition_tracker(tracker: Dict[str, Any], end_ts: pd.Timestamp) -> None:
    if tracker["active"] and tracker["current_start"] is not None:
        tracker["intervals"].append((tracker["current_start"], end_ts))
        tracker["occurrences"] += 1
        tracker["active"] = False
        tracker["current_start"] = None


def _format_interval_summary(intervals: List[Tuple[pd.Timestamp, pd.Timestamp]], limit: int = 3) -> str:
    if not intervals:
        return "-"
    formatted = [
        f"{start.strftime('%d/%m %H:%M')} → {end.strftime('%d/%m %H:%M')}"
        for start, end in intervals[:limit]
    ]
    if len(intervals) > limit:
        formatted.append(f"+{len(intervals) - limit} autres")
    return "\n".join(formatted)


def _build_interval_table(intervals: List[Tuple[pd.Timestamp, pd.Timestamp]]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for idx, (start, end) in enumerate(intervals, 1):
        duration = max(int(round((end - start).total_seconds() / 60)), 0)
        rows.append(
            {
                "Période": idx,
                "Début": start,
                "Fin": end,
                "Durée_Minutes": duration,
            }
        )
    return pd.DataFrame(rows)


def _analyze_station_conditions(
    timelines: Dict[str, pd.DataFrame],
    start_dt: datetime,
    end_dt: datetime,
) -> Dict[str, Any]:
    start_ts = _ensure_paris_timestamp(start_dt)
    end_ts = _ensure_paris_timestamp(end_dt)

    if start_ts is None or end_ts is None or end_ts <= start_ts:
        empty_df = pd.DataFrame()
        return {
            "summary_df": empty_df,
            "metrics": {
                "reference_minutes": 0,
                "downtime_minutes": 0,
                "uptime_minutes": 0,
                "availability_pct": 0.0,
                "coverage_pct": 0.0,
                "window_minutes": 0,
                "downtime_occurrences": 0,
            },
            "condition_intervals": {},
            "downtime_intervals": [],
        }

    intervals_by_equip: Dict[str, List[Tuple[pd.Timestamp, pd.Timestamp, int]]] = {}
    boundaries: Set[pd.Timestamp] = {start_ts, end_ts}

    for equip, df in timelines.items():
        equip_intervals: List[Tuple[pd.Timestamp, pd.Timestamp, int]] = []
        if df is not None and not df.empty:
            for _, row in df.iterrows():
                raw_start = _ensure_paris_timestamp(row.get("date_debut"))
                raw_end = _ensure_paris_timestamp(row.get("date_fin"))
                if raw_start is None or raw_end is None:
                    continue
                seg_start = max(raw_start, start_ts)
                seg_end = min(raw_end, end_ts)
                if seg_end <= seg_start:
                    continue
                status = int(row.get("est_disponible", -1))
                equip_intervals.append((seg_start, seg_end, status))
                boundaries.add(seg_start)
                boundaries.add(seg_end)
        equip_intervals.sort(key=lambda item: item[0])
        intervals_by_equip[equip] = equip_intervals

    if len(boundaries) <= 1:
        empty_df = pd.DataFrame()
        return {
            "summary_df": empty_df,
            "metrics": {
                "reference_minutes": 0,
                "downtime_minutes": 0,
                "uptime_minutes": 0,
                "availability_pct": 0.0,
                "coverage_pct": 0.0,
                "window_minutes": 0,
                "downtime_occurrences": 0,
            },
            "condition_intervals": {},
            "downtime_intervals": [],
        }

    ordered_boundaries = sorted(boundaries)

    def status_at(intervals: List[Tuple[pd.Timestamp, pd.Timestamp, int]], ts: pd.Timestamp) -> int:
        for start, end, status in intervals:
            if start <= ts < end:
                return status
        return -1

    condition_labels = {
        "ac_down": "Réseau AC indisponible",
        "batt_down": "DC1 & DC2 indisponibles",
        "pdc_down": "≥3 PDC indisponibles",
    }
    trackers = {key: _new_condition_tracker(label) for key, label in condition_labels.items()}

    station_tracker = {
        "duration": 0.0,
        "occurrences": 0,
        "intervals": [],
        "active": False,
        "current_start": None,
    }

    reference_minutes = 0.0
    window_minutes = max(int(round((end_ts - start_ts).total_seconds() / 60)), 0)

    pdc_names = [f"PDC{i}" for i in range(1, 7)]

    for idx in range(len(ordered_boundaries) - 1):
        seg_start = ordered_boundaries[idx]
        seg_end = ordered_boundaries[idx + 1]
        if seg_end <= seg_start:
            continue

        duration = (seg_end - seg_start).total_seconds() / 60
        if duration <= 0:
            continue

        ac_status = status_at(intervals_by_equip.get("AC", []), seg_start)
        dc1_status = status_at(intervals_by_equip.get("DC1", []), seg_start)
        dc2_status = status_at(intervals_by_equip.get("DC2", []), seg_start)
        pdc_statuses = [status_at(intervals_by_equip.get(name, []), seg_start) for name in pdc_names]

        ac_data = ac_status in (0, 1)
        batt_data = (dc1_status in (0, 1)) and (dc2_status in (0, 1))
        pdc_data = all(status in (0, 1) for status in pdc_statuses)

        ac_down = ac_data and ac_status == 0
        batt_down = batt_data and dc1_status == 0 and dc2_status == 0
        pdc_down = pdc_data and sum(status == 0 for status in pdc_statuses) >= 3

        segment_has_data = ac_data or batt_data or pdc_data

        if segment_has_data:
            reference_minutes += duration
        else:
            if station_tracker["active"]:
                station_tracker["intervals"].append((station_tracker["current_start"], seg_start))
                station_tracker["occurrences"] += 1
                station_tracker["active"] = False
                station_tracker["current_start"] = None

        _update_condition_tracker(trackers["ac_down"], ac_down, ac_data, seg_start, seg_end, duration)
        _update_condition_tracker(trackers["batt_down"], batt_down, batt_data, seg_start, seg_end, duration)
        _update_condition_tracker(trackers["pdc_down"], pdc_down, pdc_data, seg_start, seg_end, duration)

        any_condition = (
            (ac_down and ac_data)
            or (batt_down and batt_data)
            or (pdc_down and pdc_data)
        )

        if segment_has_data and any_condition:
            station_tracker["duration"] += duration
            if not station_tracker["active"]:
                station_tracker["active"] = True
                station_tracker["current_start"] = seg_start
        else:
            if station_tracker["active"]:
                station_tracker["intervals"].append((station_tracker["current_start"], seg_start))
                station_tracker["occurrences"] += 1
                station_tracker["active"] = False
                station_tracker["current_start"] = None

    for tracker in trackers.values():
        _finalize_condition_tracker(tracker, end_ts)

    if station_tracker["active"] and station_tracker["current_start"] is not None:
        station_tracker["intervals"].append((station_tracker["current_start"], end_ts))
        station_tracker["occurrences"] += 1
        station_tracker["active"] = False
        station_tracker["current_start"] = None

    reference_minutes_int = max(int(round(reference_minutes)), 0)
    downtime_minutes_int = max(int(round(station_tracker["duration"])), 0)
    uptime_minutes_int = max(reference_minutes_int - downtime_minutes_int, 0)

    availability_pct = (uptime_minutes_int / reference_minutes_int * 100) if reference_minutes_int > 0 else 0.0
    coverage_pct = (reference_minutes_int / window_minutes * 100) if window_minutes > 0 else 0.0

    summary_rows: List[Dict[str, Any]] = []
    condition_intervals: Dict[str, List[Tuple[pd.Timestamp, pd.Timestamp]]] = {}

    for tracker in trackers.values():
        duration_int = max(int(round(tracker["duration"])), 0)
        analyzed_int = max(int(round(tracker["denom"])), 0)
        pct_condition = (duration_int / analyzed_int * 100) if analyzed_int > 0 else 0.0
        pct_station = (duration_int / reference_minutes_int * 100) if reference_minutes_int > 0 else 0.0
        coverage_condition = (analyzed_int / window_minutes * 100) if window_minutes > 0 else 0.0

        summary_rows.append(
            {
                "Condition": tracker["label"],
                "Occurrences": tracker["occurrences"],
                "Durée_Minutes": duration_int,
                "Temps_Analysé_Minutes": analyzed_int,
                "Part_Temps_Analysé": round(pct_condition, 2),
                "Part_Temps_Station": round(pct_station, 2),
                "Couverture_Période": round(coverage_condition, 1),
                "Périodes_Clés": _format_interval_summary(tracker["intervals"]),
            }
        )
        condition_intervals[tracker["label"]] = tracker["intervals"]

    summary_df = pd.DataFrame(summary_rows)

    return {
        "summary_df": summary_df,
        "metrics": {
            "reference_minutes": reference_minutes_int,
            "downtime_minutes": downtime_minutes_int,
            "uptime_minutes": uptime_minutes_int,
            "availability_pct": round(availability_pct, 2),
            "coverage_pct": round(coverage_pct, 1),
            "window_minutes": window_minutes,
            "downtime_occurrences": station_tracker["occurrences"],
        },
        "condition_intervals": condition_intervals,
        "downtime_intervals": station_tracker["intervals"],
    }


@st.cache_data(ttl=900, show_spinner=False)
def load_station_statistics(site: str, start_dt: datetime, end_dt: datetime) -> Dict[str, Any]:
    timelines: Dict[str, pd.DataFrame] = {}

    for equip, mode in _station_equipment_modes():
        try:
            df = load_blocks(site, equip, start_dt, end_dt, mode=mode)
        except Exception as exc:
            logger.error("Erreur lors du chargement de %s pour %s : %s", equip, site, exc)
            df = pd.DataFrame()
        timelines[equip] = df.copy() if df is not None and not df.empty else pd.DataFrame()

    analysis = _analyze_station_conditions(timelines, start_dt, end_dt)
    analysis["timeline_df"] = _build_station_timeline_df(timelines)
    return analysis


@st.cache_data(ttl=1800, show_spinner=False)
def _calculate_monthly_availability_equipment(
    site: Optional[str] = None,
    equip: Optional[str] = None,
    months: int = 12,
    start_dt: Optional[datetime] = None,
    end_dt: Optional[datetime] = None,
) -> pd.DataFrame:
    if not start_dt or not end_dt:
        end_dt = datetime.utcnow()
        start_dt = (end_dt.replace(day=1) - pd.DateOffset(months=months)).to_pydatetime()

    df = load_filtered_blocks(start_dt, end_dt, site, equip, mode=MODE_EQUIPMENT)
    if df.empty:
        return df

    return _aggregate_monthly_availability(df, start_dt, end_dt)



@st.cache_data(ttl=1800, show_spinner=False)
def _calculate_monthly_availability_pdc(
    site: Optional[str] = None,
    equip: Optional[str] = None,
    months: int = 12,
    start_dt: Optional[datetime] = None,
    end_dt: Optional[datetime] = None,
) -> pd.DataFrame:
    if not start_dt or not end_dt:
        end_dt = datetime.utcnow()
        start_dt = (end_dt.replace(day=1) - pd.DateOffset(months=months)).to_pydatetime()

    df = load_filtered_blocks(start_dt, end_dt, site, equip, mode=MODE_PDC)
    if df.empty:
        return df

    return _aggregate_monthly_availability(df, start_dt, end_dt)



def calculate_monthly_availability(
    site: Optional[str] = None,
    equip: Optional[str] = None,
    months: int = 12,
    start_dt: Optional[datetime] = None,
    end_dt: Optional[datetime] = None,
    mode: Optional[str] = None,
) -> pd.DataFrame:
    active_mode = mode or get_current_mode()
    if active_mode == MODE_PDC:
        return _calculate_monthly_availability_pdc(site, equip, months, start_dt, end_dt)
    return _calculate_monthly_availability_equipment(site, equip, months, start_dt, end_dt)

def get_unavailability_causes(df: Optional[pd.DataFrame]) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()

    causes = (
        df.loc[df["est_disponible"] == 0]
        .groupby("cause", dropna=False)["duration_minutes"]
        .sum()
        .reset_index()
        .sort_values("duration_minutes", ascending=False)
    )
    
    if not causes.empty:
        causes["percentage"] = (causes["duration_minutes"] / causes["duration_minutes"].sum() * 100)
        causes["cause"] = causes["cause"].fillna("Non spécifié")
    
    return causes

def translate_cause_to_text(cause: str, equipement_id: str) -> str:
    if not cause or cause == "Non spécifié":
        return "Cause non spécifiée"
    try:
        ic_val: Optional[int] = None
        pc_val: Optional[int] = None

        normalized = cause.replace("\n", " ")
        pattern = re.compile(r"\b((?:IC|PC)1?)\s*[:=]?\s*(-?\d+)", re.IGNORECASE)

        for key, value in pattern.findall(normalized):
            key_upper = key.upper()
            parsed_value = int(value)
            if key_upper.startswith("IC") and ic_val is None:
                ic_val = parsed_value
            elif key_upper.startswith("PC") and pc_val is None:
                pc_val = parsed_value

        if ic_val is None or pc_val is None:
            numbers = re.findall(r"-?\d+", normalized)
            if numbers:
                if ic_val is None and len(numbers) >= 1:
                    ic_val = int(numbers[0])
                if pc_val is None and len(numbers) >= 2:
                    pc_val = int(numbers[1])

        
        if ic_val is not None or pc_val is not None:
            cfg = get_equip_config(equipement_id)
            translated = translate_ic_pc(ic_val, pc_val, cfg["ic_map"], cfg["pc_map"])
            return translated if translated else cause
        
        return cause
        
    except Exception:
        return cause

def get_translated_unavailability_causes(df: Optional[pd.DataFrame], equipement_id: str) -> pd.DataFrame:

    if df is None or df.empty:
        return pd.DataFrame()

    unavailable_data = df.loc[df["est_disponible"] == 0].copy()

    if unavailable_data.empty:
        return pd.DataFrame()
    
    unavailable_data["cause_translated"] = unavailable_data["cause"].apply(
        lambda x: translate_cause_to_text(x, equipement_id)
    )
    
    causes = (
        unavailable_data
        .groupby("cause_translated", dropna=False)["duration_minutes"]
        .sum()
        .reset_index()
        .sort_values("duration_minutes", ascending=False)
    )
    
    if not causes.empty:
        causes["percentage"] = (causes["duration_minutes"] / causes["duration_minutes"].sum() * 100)
        causes["cause_translated"] = causes["cause_translated"].fillna("Cause non spécifiée")
    
    return causes.rename(columns={"cause_translated": "cause"})

@st.cache_data(ttl=1800, show_spinner=False)
def get_equipment_summary(
    start_dt: datetime,
    end_dt: datetime,
    site: Optional[str] = None,
    mode: Optional[str] = None,
) -> pd.DataFrame:
    """Génère un tableau récapitulatif des équipements pour le mode actif."""
    active_mode = mode or get_current_mode()
    equipments = get_equipments(active_mode, site)
    if not equipments:
        return pd.DataFrame(columns=[
            "Équipement",
            "Disponibilité Brute (%)",
            "Disponibilité Avec Exclusions (%)",
            "Durée Totale",
            "Temps Disponible",
            "Temps Indisponible",
            "Jours avec des données",
        ])

    df = load_filtered_blocks(start_dt, end_dt, site, None, mode=active_mode)
    if df.empty:
        return pd.DataFrame([
            {
                "Équipement": equip,
                "Disponibilité Brute (%)": 0.0,
                "Disponibilité Avec Exclusions (%)": 0.0,
                "Durée Totale": "0 minutes",
                "Temps Disponible": "0 minutes",
                "Temps Indisponible": "0 minutes",
                "Jours avec des données": 0,
            }
            for equip in equipments
        ])

    summary_rows = []
    for equip in equipments:
        equip_data = df[df["equipement_id"] == equip]
        if equip_data.empty:
            summary_rows.append({
                "Équipement": equip,
                "Disponibilité Brute (%)": 0.0,
                "Disponibilité Avec Exclusions (%)": 0.0,
                "Durée Totale": "0 minutes",
                "Temps Disponible": "0 minutes",
                "Temps Indisponible": "0 minutes",
                "Jours avec des données": 0,
            })
            continue

        stats_raw = calculate_availability(equip_data, include_exclusions=False)
        stats_excl = calculate_availability(equip_data, include_exclusions=True)
        days_with_data = (
            pd.to_datetime(equip_data["date_debut"]).dt.floor("D").nunique()
        )
        summary_rows.append({
            "Équipement": equip,
            "Disponibilité Brute (%)": round(stats_raw["pct_available"], 2),
            "Disponibilité Avec Exclusions (%)": round(stats_excl["pct_available"], 2),
            "Durée Totale": format_minutes(stats_raw["total_minutes"]),
            "Temps Disponible": format_minutes(stats_raw["available_minutes"]),
            "Temps Indisponible": format_minutes(stats_raw["unavailable_minutes"]),
            "Jours avec des données": int(days_with_data),
        })

    if active_mode == MODE_PDC and not df.empty:
        global_stats_raw = calculate_availability(df, include_exclusions=False)
        global_stats_excl = calculate_availability(df, include_exclusions=True)
        global_days = (
            pd.to_datetime(df["date_debut"]).dt.floor("D").nunique()
        )
        if site:
            label = "Dispo globale site"
        else:
            label = "Dispo globale (tous sites)"
        global_row = {
            "Équipement": label,
            "Disponibilité Brute (%)": round(global_stats_raw["pct_available"], 2),
            "Disponibilité Avec Exclusions (%)": round(global_stats_excl["pct_available"], 2),
            "Durée Totale": format_minutes(global_stats_raw["total_minutes"]),
            "Temps Disponible": format_minutes(global_stats_raw["available_minutes"]),
            "Temps Indisponible": format_minutes(global_stats_raw["unavailable_minutes"]),
            "Jours avec des données": int(global_days),
        }
        summary_rows = [global_row] + summary_rows

    return pd.DataFrame(summary_rows)

@st.cache_data(ttl=1800, show_spinner=False)
def generate_availability_report(
    start_dt: datetime,
    end_dt: datetime,
    site: Optional[str] = None,
    mode: Optional[str] = None,
) -> Dict[str, pd.DataFrame]:
    """Génère un rapport complet de disponibilité pour tous les équipements."""
    active_mode = mode or get_current_mode()
    equipments = get_equipments(active_mode, site)
    if not equipments:
        return {}

    params = {"start": start_dt, "end": end_dt}
    if active_mode == MODE_PDC:
        if site:
            union_sql = _pdc_union_sql_for_site(site)
            params["site"] = site
            site_filter = "AND b.site = :site"
        else:
            union_sql = _pdc_union_sql_all_sites()
            site_filter = ""
        q = f"""
        WITH base AS (
            {union_sql}
        )
        SELECT
          b.bloc_id,
          b.site, b.equipement_id, b.date_debut, b.date_fin, b.est_disponible, b.cause,
          TIMESTAMPDIFF(MINUTE, GREATEST(b.date_debut,:start), LEAST(b.date_fin,:end)) AS duration_minutes,
          COALESCE(e.previous_status, b.est_disponible) AS previous_status,
          CASE WHEN e.id IS NOT NULL THEN 1 ELSE 0 END AS is_excluded,
          e.id AS exclusion_id,
          e.applied_by AS exclusion_applied_by,
          e.applied_at AS exclusion_applied_at,
          e.exclusion_comment AS exclusion_comment,
          b.source_table
        FROM base b
        LEFT JOIN dispo_blocs_exclusions e
          ON e.table_name = b.source_table
         AND e.bloc_id = b.bloc_id
         AND e.released_at IS NULL
        WHERE b.date_debut < :end AND b.date_fin > :start
          {site_filter}
        ORDER BY b.equipement_id, b.date_debut
        """
    else:
        if site:
            ac_union = _ac_union_sql_for_site(site)
            batt_union = _batt_union_sql_for_site(site)
            params["site"] = site
            site_filter_ac = "WHERE site = :site"
            site_filter_bt = "WHERE site = :site"
        else:
            ac_union = _ac_union_sql_all_sites()
            batt_union = _batt_union_sql_all_sites()
            site_filter_ac = ""
            site_filter_bt = ""

        q = f"""
        WITH ac AS (
            {ac_union}
        ),
        batt AS (
            {batt_union}
        ),
        base AS (
            SELECT
              bloc_id, source_table,
              site, equipement_id, type_equipement, date_debut, date_fin,
              est_disponible, cause, raw_point_count, processed_at, batch_id, hash_signature
            FROM ac {site_filter_ac}
            UNION ALL
            SELECT
              bloc_id, source_table,
              site, equipement_id, type_equipement, date_debut, date_fin,
              est_disponible, cause, raw_point_count, processed_at, batch_id, hash_signature
            FROM batt {site_filter_bt}
        )
        SELECT
          b.bloc_id,
          b.site, b.equipement_id, b.date_debut, b.date_fin, b.est_disponible, b.cause,
          TIMESTAMPDIFF(MINUTE, GREATEST(b.date_debut,:start), LEAST(b.date_fin,:end)) AS duration_minutes,
          COALESCE(e.previous_status, b.est_disponible) AS previous_status,
          CASE WHEN e.id IS NOT NULL THEN 1 ELSE 0 END AS is_excluded,
          e.id AS exclusion_id,
          e.applied_by AS exclusion_applied_by,
          e.applied_at AS exclusion_applied_at,
          e.exclusion_comment AS exclusion_comment,
          b.source_table
        FROM base b
        LEFT JOIN dispo_blocs_exclusions e
          ON e.table_name = b.source_table
         AND e.bloc_id = b.bloc_id
         AND e.released_at IS NULL
        WHERE b.date_debut < :end AND b.date_fin > :start
        ORDER BY b.equipement_id, b.date_debut
        """

    df = execute_query(q, params)
    df = _normalize_blocks_df(df)

    if df.empty:
        return {}

    report: Dict[str, pd.DataFrame] = {}

    for equip in equipments:
        equip_data = df[df["equipement_id"] == equip]

        if equip_data.empty:
            report[equip] = pd.DataFrame(columns=[
                "ID", "Site", "Équipement", "Début", "Fin", "Durée",
                "Statut", "Cause Originale", "Cause Traduite", "Exclu"
            ])
            continue

        stats_raw = calculate_availability(equip_data, include_exclusions=False)
        stats_excl = calculate_availability(equip_data, include_exclusions=True)

        report_data = []
        report_data.append({
            "ID": "RÉSUMÉ",
            "Site": equip_data["site"].iloc[0] if not equip_data.empty else "N/A",
            "Équipement": equip,
            "Début": start_dt.strftime("%Y-%m-%d %H:%M"),
            "Fin": end_dt.strftime("%Y-%m-%d %H:%M"),
            "Durée": format_minutes(stats_raw["total_minutes"]),
            "Durée_Minutes": stats_raw["total_minutes"],
            "Statut": f"Disponibilité: {stats_raw['pct_available']:.2f}%",
            "Cause Originale": f"Brute: {stats_raw['pct_available']:.2f}% | Avec exclusions: {stats_excl['pct_available']:.2f}%",
            "Cause Traduite": f"Disponible: {format_minutes(stats_raw['available_minutes'])} | Indisponible: {format_minutes(stats_raw['unavailable_minutes'])}",
            "Exclu": "N/A",
        })

        unavailable_blocks = equip_data[equip_data["est_disponible"] == 0].copy()
        for idx, (_, block) in enumerate(unavailable_blocks.iterrows(), 1):
            cause_originale = block.get("cause", "Non spécifié")
            cause_traduite = translate_cause_to_text(cause_originale, equip)
            report_data.append({
                "ID": f"IND-{idx:03d}",
                "Site": block["site"],
                "Équipement": equip,
                "Début": block["date_debut"].strftime("%Y-%m-%d %H:%M"),
                "Fin": block["date_fin"].strftime("%Y-%m-%d %H:%M"),
                "Durée": format_minutes(int(block["duration_minutes"])),
                "Durée_Minutes": int(block["duration_minutes"]),
                "Statut": "❌ Indisponible",
                "Cause Originale": cause_originale,
                "Cause Traduite": cause_traduite,
                "Exclu": "✅ Oui" if block["is_excluded"] == 1 else "❌ Non",
            })

        missing_blocks = equip_data[equip_data["est_disponible"] == -1].copy()
        for idx, (_, block) in enumerate(missing_blocks.iterrows(), 1):
            report_data.append({
                "ID": f"MISS-{idx:03d}",
                "Site": block["site"],
                "Équipement": equip,
                "Début": block["date_debut"].strftime("%Y-%m-%d %H:%M"),
                "Fin": block["date_fin"].strftime("%Y-%m-%d %H:%M"),
                "Durée": format_minutes(int(block["duration_minutes"])),
                "Durée_Minutes": int(block["duration_minutes"]),
                "Statut": "⚠️ Données manquantes",
                "Cause Originale": "Données manquantes",
                "Cause Traduite": "Aucune donnée disponible pour cette période",
                "Exclu": "✅ Oui" if block["is_excluded"] == 1 else "❌ Non",
            })

        report[equip] = pd.DataFrame(report_data)

    return report

def analyze_daily_unavailability(unavailable_data: pd.DataFrame) -> pd.DataFrame:
    """Analyse les indisponibilités par jour."""
    if unavailable_data.empty:
        return pd.DataFrame()
    
    # Convertir les dates en datetime si nécessaire
    unavailable_data = unavailable_data.copy()
    unavailable_data["date_debut"] = pd.to_datetime(unavailable_data["date_debut"])
    unavailable_data["date_fin"] = pd.to_datetime(unavailable_data["date_fin"])
    
    # Extraire la date (sans l'heure) pour le groupement
    unavailable_data["date_jour"] = unavailable_data["date_debut"].dt.date
    
    # Grouper par jour et calculer les statistiques
    daily_stats = []
    
    for date_jour, group in unavailable_data.groupby("date_jour"):
        # Compter le nombre de périodes d'indisponibilité
        nb_periodes = len(group)
        
        # Calculer la durée totale d'indisponibilité pour ce jour
        duree_totale_minutes = group["Durée_Minutes"].sum()
        
        # Trouver la première et dernière heure d'indisponibilité
        heure_debut = group["date_debut"].min().strftime("%H:%M")
        heure_fin = group["date_fin"].max().strftime("%H:%M")
        
        # Calculer le pourcentage de la journée en indisponibilité
        # Supposons une journée de 24h = 1440 minutes
        pourcentage_journee = (duree_totale_minutes / 1440) * 100
        
        # Traduire le nom du jour en français
        jours_fr = {
            'Monday': 'Lundi', 'Tuesday': 'Mardi', 'Wednesday': 'Mercredi',
            'Thursday': 'Jeudi', 'Friday': 'Vendredi', 'Saturday': 'Samedi', 'Sunday': 'Dimanche'
        }
        jour_nom = jours_fr.get(date_jour.strftime("%A"), date_jour.strftime("%A"))
        
        daily_stats.append({
            "Date": date_jour.strftime("%Y-%m-%d"),
            "Jour": jour_nom,
            "Nb Périodes": nb_periodes,
            "Durée Totale": format_minutes(duree_totale_minutes),
            "Durée_Minutes": duree_totale_minutes,  # Pour le tri
            "Première Heure": heure_debut,
            "Dernière Heure": heure_fin,
            "% Journée": f"{pourcentage_journee:.1f}%"
        })
    
    # Trier par date décroissante (plus récent en premier)
    daily_df = pd.DataFrame(daily_stats)
    if not daily_df.empty:
        daily_df = daily_df.sort_values("Date", ascending=False)
    
    return daily_df

def analyze_daily_unavailability_by_equipment(unavailable_data: pd.DataFrame) -> pd.DataFrame:
    """Analyse les indisponibilités par jour et par équipement."""
    if unavailable_data.empty:
        return pd.DataFrame()
    
    # Convertir les dates en datetime si nécessaire
    unavailable_data = unavailable_data.copy()
    unavailable_data["date_debut"] = pd.to_datetime(unavailable_data["date_debut"])
    unavailable_data["date_fin"] = pd.to_datetime(unavailable_data["date_fin"])
    
    # Extraire la date (sans l'heure) pour le groupement
    unavailable_data["date_jour"] = unavailable_data["date_debut"].dt.date
    
    # Grouper par jour et équipement
    daily_stats = []
    
    for (date_jour, equip), group in unavailable_data.groupby(["date_jour", "Équipement"]):
        # Compter le nombre de périodes d'indisponibilité
        nb_periodes = len(group)
        
        # Calculer la durée totale d'indisponibilité pour ce jour et cet équipement
        duree_totale_minutes = group["Durée_Minutes"].sum()
        
        # Trouver la première et dernière heure d'indisponibilité
        heure_debut = group["date_debut"].min().strftime("%H:%M")
        heure_fin = group["date_fin"].max().strftime("%H:%M")
        
        # Calculer le pourcentage de la journée en indisponibilité
        # Supposons une journée de 24h = 1440 minutes
        pourcentage_journee = (duree_totale_minutes / 1440) * 100
        
        # Traduire le nom du jour en français
        jours_fr = {
            'Monday': 'Lundi', 'Tuesday': 'Mardi', 'Wednesday': 'Mercredi',
            'Thursday': 'Jeudi', 'Friday': 'Vendredi', 'Saturday': 'Samedi', 'Sunday': 'Dimanche'
        }
        jour_nom = jours_fr.get(date_jour.strftime("%A"), date_jour.strftime("%A"))
        
        daily_stats.append({
            "Date": date_jour.strftime("%Y-%m-%d"),
            "Jour": jour_nom,
            "Équipement": equip,
            "Nb Périodes": nb_periodes,
            "Durée Totale": format_minutes(duree_totale_minutes),
            "Durée_Minutes": duree_totale_minutes,  # Pour le tri
            "Première Heure": heure_debut,
            "Dernière Heure": heure_fin,
            "% Journée": f"{pourcentage_journee:.1f}%"
        })
    
    # Trier par date décroissante puis par durée décroissante
    daily_df = pd.DataFrame(daily_stats)
    if not daily_df.empty:
        daily_df = daily_df.sort_values(["Date", "Durée_Minutes"], ascending=[False, False])
    
    return daily_df

# ui
def format_minutes(total_minutes: int) -> str:
    """Formate en 'X jours, Y heures, Z minutes' (avec pluriels corrects)."""
    m = int(total_minutes or 0)
    days, rem = divmod(m, 1440)   # 1440 = 24*60
    hours, mins = divmod(rem, 60)

    parts = []
    if days:
        parts.append(f"{days} {'jour' if days == 1 else 'jours'}")
    if hours:
        parts.append(f"{hours} {'heure' if hours == 1 else 'heures'}")
    if mins or not parts:
        parts.append(f"{mins} {'minute' if mins == 1 else 'minutes'}")

    return ", ".join(parts)

def render_header():
    """Affiche l'en-tête de l'application."""
    col1, col2, col3 = st.columns([3, 2, 1])
    with col1:
        st.title("📊 Tableau de Bord - Disponibilité des Équipements")
        st.caption("Analyse et suivi de la disponibilité opérationnelle")
    with col2:
        options = [MODE_EQUIPMENT, MODE_PDC]
        current_mode = get_current_mode()
        index = options.index(current_mode) if current_mode in options else 0
        selected_mode = st.radio(
            "Mode d'analyse",
            options=options,
            index=index,
            horizontal=True,
            format_func=lambda k: MODE_LABELS.get(k, k),
            help="Basculer entre la disponibilité des équipements et celle des points de charge",
        )
        if selected_mode != current_mode:
            set_current_mode(selected_mode)
    with col3:
        if st.button("🔄 Actualiser", use_container_width=True):
            invalidate_cache()
            st.rerun()
def render_filters() -> Tuple[Optional[str], Optional[str], datetime, datetime]:
    """Affiche les filtres et retourne les valeurs sélectionnées."""
    mode = get_current_mode()
    st.subheader("🔍 Filtres de Recherche")

    col1, col2, col3 = st.columns([1, 1, 2])
    with col1:
        site_codes = get_sites(mode) or []
        if not site_codes:
            st.warning("Aucun site disponible.")
            return None, None, datetime.min, datetime.min  
        selected_site = st.selectbox(
            "Site",
            options=site_codes,               
            index=0,
            format_func=lambda code: mapping_sites.get(code.split("_")[-1], code),
            help="Sélectionnez un site"
        )
        site = selected_site

    with col2:
        equips = get_equipments(mode, site) if site else get_equipments(mode)
        equips = equips or []
        if not equips:
            st.warning("Aucun équipement pour ce site.")
            return site, None, datetime.min, datetime.min  

        selected_equip = st.selectbox(
            "Équipement",
            options=equips,                    
            index=0,
            format_func=lambda value: value,
            help="Sélectionnez un équipement"
        )
        equip = selected_equip

    with col3:
        today = datetime.now(timezone.utc).date()
        c1, c2 = st.columns(2)
        
        default_start = st.session_state.get("filter_start_date", today - timedelta(days=30))
        start_date = c1.date_input(
            "Date de début",
            value=default_start,
            max_value=today,
            key="filter_start_date",
            help="Date de début de la période d'analyse"
        )

        default_end = st.session_state.get("filter_end_date", today)
        if isinstance(default_end, datetime):
            default_end = default_end.date()
        if default_end < start_date:
            default_end = start_date

        end_date = c2.date_input(
            "Date de fin",
            value=default_end,
            min_value=start_date,
            max_value=today,
            key="filter_end_date",
            help="Date de fin de la période d'analyse"
        )

        if end_date < start_date:
            st.session_state["filter_end_date"] = start_date
            end_date = start_date
    
    start_dt = datetime.combine(start_date, time.min)
    end_dt = datetime.combine(end_date, time.max)

    return site, equip, start_dt, end_dt

def render_overview_tab(df: Optional[pd.DataFrame]):
    """Affiche l'onglet vue d'ensemble."""
    mode = get_current_mode()
    st.header("📈 Vue d'Ensemble")

    site_scope = st.session_state.get("current_site")
    equip_scope = st.session_state.get("current_equip")
    context_parts = []
    if site_scope is None:
        context_parts.append("tous les sites")
    if equip_scope is None:
        if mode == MODE_PDC:
            context_parts.append("l'ensemble des points de charge")
        else:
            context_parts.append("l'ensemble des équipements")
    if context_parts:
        st.info("Vue générale : " + " et ".join(context_parts) + ".")

    if df is None or df.empty:
        st.warning("⚠️ Aucune donnée disponible pour les critères sélectionnés.")
        st.info("💡 Conseil: Essayez d'élargir la période ou de modifier les filtres.")
        return

    stats_raw = calculate_availability(df, include_exclusions=False)
    stats_excl = calculate_availability(df, include_exclusions=True)

    st.subheader("📊 Indicateurs Clés")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Disponibilité brute",
            f"{stats_raw['pct_available']:.2f}%",
            help="Valeur correspondant au calcul standard"
        )

    with col2:
        st.metric(
            "Disponibilité avec exclusions",
            f"{stats_excl['pct_available']:.2f}%",
            delta=f"{stats_excl['pct_available'] - stats_raw['pct_available']:.2f}%",
            help="Différence par rapport au calcul brut"
        )

    with col3:
        analyzed_minutes = stats_excl['effective_minutes']
        analyzed_delta = analyzed_minutes - stats_raw['effective_minutes']
        if analyzed_delta:
            delta_prefix = "+" if analyzed_delta > 0 else "-"
            delta_value = f"{delta_prefix} {format_minutes(abs(analyzed_delta))}"
        else:
            delta_value = None

        st.metric(
            "Temps analysé",
            format_minutes(analyzed_minutes),
            delta=delta_value,
            help=(
                "Temps total disposant de données après application des exclusions "
                f"(données manquantes initiales : {format_minutes(stats_raw['missing_minutes'])})."
            )
        )

    with col4:
        st.metric(
            "Temps Indisponible (avec exclusions)",
            format_minutes(stats_excl['unavailable_minutes']),
            delta=f"{stats_excl['unavailable_minutes'] - stats_raw['unavailable_minutes']} min",
            delta_color="inverse", 
            help="Temps total d'indisponibilité après application des exclusions"
        )
    st.divider()
    
    st.subheader("📋 Tableau Récapitulatif des Équipements")
    
    site_current = st.session_state.get("current_site")
    start_dt_current = st.session_state.get("current_start_dt")
    end_dt_current = st.session_state.get("current_end_dt")
    
    if start_dt_current and end_dt_current:
        df_summary = get_equipment_summary(start_dt_current, end_dt_current, site_current, mode=mode)
        
        if not df_summary.empty:
            # Afficher le tableau avec un style amélioré
            st.dataframe(
                df_summary,
                hide_index=True,
                use_container_width=True,
                column_config={
                    "Équipement": st.column_config.TextColumn("Équipement", width="medium"),
                    "Disponibilité Brute (%)": st.column_config.NumberColumn(
                        "Disponibilité Brute (%)",
                        width="medium",
                        format="%.2f%%"
                    ),
                    "Disponibilité Avec Exclusions (%)": st.column_config.NumberColumn(
                        "Disponibilité Avec Exclusions (%)",
                        width="medium",
                        format="%.2f%%"
                    ),
                    "Durée Totale": st.column_config.TextColumn("Durée Totale", width="medium"),
                    "Temps Disponible": st.column_config.TextColumn("Temps Disponible", width="medium"),
                    "Temps Indisponible": st.column_config.TextColumn("Temps Indisponible", width="medium"),
                    "Jours avec des données": st.column_config.NumberColumn(
                        "Jours avec des données",
                        width="small"
                    )
                }
            )
            
            # Ajouter des métriques visuelles pour chaque équipement
            col1, col2, col3 = st.columns(3)
            column_cycle = cycle([col1, col2, col3])

            for _, row in df_summary.iterrows():
                with next(column_cycle):
                    equip = row["Équipement"]
                    pct_brut = row["Disponibilité Brute (%)"]
                    pct_excl = row["Disponibilité Avec Exclusions (%)"]

                    # Couleur selon la disponibilité
                    if pct_brut >= 95:
                        color = "normal"
                    elif pct_brut >= 90:
                        color = "off"
                    else:
                        color = "inverse"
                    
                    st.metric(
                        f"{equip} - Disponibilité",
                        f"{pct_brut:.2f}%",
                        delta=f"{pct_excl - pct_brut:.2f}%",
                        delta_color=color,
                        help=f"Brute: {pct_brut:.2f}% | Avec exclusions: {pct_excl:.2f}%"
                    )
        else:
            st.info("ℹ️ Aucune donnée disponible pour le tableau récapitulatif.")
    else:
        st.warning("⚠️ Impossible de générer le tableau récapitulatif sans période définie.")
    
    st.divider()
    
    st.subheader("🔍 Analyse des Indisponibilités")
    causes = get_unavailability_causes(df)

    if causes.empty:
        st.success("Aucune indisponibilité détectée sur la période")
    else:
        color_map = px.colors.qualitative.Safe  
        unique_causes = causes["cause"].unique()
        cause_colors = {cause: color_map[i % len(color_map)] for i, cause in enumerate(unique_causes)}
        
        col_chart, col_table = st.columns([2, 1])
        with col_chart:
            small_mask = causes["percentage"] < 2.5

            fig = px.pie(
                causes,
                names="cause",
                values="duration_minutes",
                title="Répartition des Causes d'Indisponibilité",
                hole=0.4,
                color="cause",
                color_discrete_map=cause_colors
            )

            fig.update_traces(
                textinfo="percent+label",
                textposition=[
                    "outside" if small else "inside" 
                    for small in small_mask
                ],
                pull=[
                    0.05 if small else 0 
                    for small in small_mask
                ],
                showlegend=True
            )
            fig.update_layout(
                uniformtext_minsize=10,
                uniformtext_mode="hide"
            )

            st.plotly_chart(fig, use_container_width=True)

        with col_table:
            df_display = causes.rename(
                columns={"duration_minutes": "Durée", "percentage": "Pourcentage"}
            )
            st.dataframe(
                df_display.style.format({
                    "Durée": lambda x: format_minutes(int(x)),
                    "Pourcentage": "{:.1f}%"
                }),
                hide_index=True,
                use_container_width=True
            )
    
    # Tableau traduit des causes d'indisponibilité
    st.subheader("📋 Causes d'Indisponibilité Traduites")
    
    # Récupérer l'équipement sélectionné pour la traduction
    current_equip = st.session_state.get("current_equip")
    
    if current_equip and not df.empty:
        # Générer le tableau traduit
        causes_translated = get_translated_unavailability_causes(df, current_equip)
        
        if not causes_translated.empty:
            st.info(f"🔧 Traduction des codes IC/PC pour l'équipement **{current_equip}**")
            
            # Afficher le tableau traduit avec un style amélioré
            df_translated_display = causes_translated.rename(
                columns={
                    "cause": "Cause Traduite", 
                    "duration_minutes": "Durée", 
                    "percentage": "Pourcentage"
                }
            )
            
            st.dataframe(
                df_translated_display.style.format({
                    "Durée": lambda x: format_minutes(int(x)),
                    "Pourcentage": "{:.1f}%"
                }),
                hide_index=True,
                use_container_width=True,
                column_config={
                    "Cause Traduite": st.column_config.TextColumn(
                        "Cause Traduite", 
                        width="large",
                        help="Description détaillée de la cause d'indisponibilité"
                    ),
                    "Durée": st.column_config.TextColumn("Durée", width="medium"),
                    "Pourcentage": st.column_config.NumberColumn("Pourcentage", width="small", format="%.1f%%")
                }
            )
            
            # Ajouter un expander avec des informations sur la traduction
            with st.expander("ℹ️ Informations sur la traduction"):
                st.markdown("""
                **Comment fonctionne la traduction :**
                
                - Les codes IC (Input Condition) et PC (Process Condition) sont extraits des causes d'indisponibilité
                - Chaque code est traduit selon la configuration de l'équipement :
                  - **AC** : SEQ01.OLI.A.IC1 / SEQ01.OLI.A.PC1
                  - **DC1** : SEQ02.OLI.A.IC1 / SEQ02.OLI.A.PC1
                  - **DC2** : SEQ03.OLI.A.IC1 / SEQ03.OLI.A.PC1
                  - **PDC** : SEQ1x/SEQ2x selon le point de charge (ex. SEQ12, SEQ22, SEQ13…)
                - Les descriptions détaillées incluent les références matérielles et les conditions de défaut
                """)
                
                # Afficher la configuration de l'équipement
                cfg = get_equip_config(current_equip)
                st.markdown(f"""
                **Configuration actuelle ({current_equip}) :**
                - Champ IC : `{cfg['ic_field']}`
                - Champ PC : `{cfg['pc_field']}`
                - Titre : {cfg['title']}
                """)
        else:
            st.info("ℹ️ Aucune cause d'indisponibilité à traduire pour cet équipement.")
    else:
        if not current_equip:
            st.warning("⚠️ Veuillez sélectionner un équipement spécifique pour voir les causes traduites.")
        else:
            st.info("ℹ️ Aucune donnée disponible pour la traduction des causes.")

    st.divider()

    
    # evolution mensuelle
    st.subheader("📅 Évolution Mensuelle")
    site = st.session_state.get("current_site")
    equip = st.session_state.get("current_equip")
    start_dt = st.session_state.get("current_start_dt")
    end_dt = st.session_state.get("current_end_dt")

    df_monthly = calculate_monthly_availability(site, equip, months=12, start_dt=start_dt, end_dt=end_dt, mode=mode)
    if not df_monthly.empty:
        months_series = pd.to_datetime(df_monthly["month"])
        month_keys = months_series.dt.strftime("%Y-%m")             
        month_labels = months_series.dt.strftime("%b %Y")            
        month_options = list(dict(zip(month_keys, month_labels)).items())  

        default_keys = list(dict.fromkeys(month_keys)) 

        sel_keys = st.multiselect(
            "Mois à afficher",
            options=[k for k, _ in month_options],
            format_func=lambda k: dict(month_options)[k],
            default=default_keys
        )

        df_monthly = df_monthly[month_keys.isin(sel_keys)].copy()
        df_monthly = df_monthly.sort_values("month")
    if df_monthly.empty:
        st.info("ℹ️ Données mensuelles insuffisantes pour l'affichage.")
    else:
        brut = df_monthly["pct_brut"].astype(float).where(pd.notna(df_monthly["pct_brut"]), None)
        excl = df_monthly["pct_excl"].astype(float).where(pd.notna(df_monthly["pct_excl"]), None)

        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=df_monthly["month"], y=brut, name="Brute",
            text=[f"{v:.1f}%" if v is not None else "" for v in brut],
            textposition="outside",
        ))
        fig.add_trace(go.Bar(
            x=df_monthly["month"], y=excl, name="Avec exclusions",
            text=[f"{v:.1f}%" if v is not None else "" for v in excl],
            textposition="outside",
        ))

        fig.update_layout(
            title="Disponibilité mensuelle",
            xaxis_title="Mois",
            yaxis_title="Disponibilité (%)",
            yaxis=dict(range=[0, 105]),
            barmode="group",
            bargap=0.25,
            hovermode="x",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        fig.update_xaxes(tickformat="%b %Y")

        st.plotly_chart(fig, use_container_width=True)
        with st.expander("📊 Statistiques détaillées"):
            df_display = df_monthly.copy()
            try:
                mois_labels = pd.to_datetime(df_display["month"]).dt.month_name(locale="fr_FR").str.capitalize() \
                            + " " + pd.to_datetime(df_display["month"]).dt.year.astype(str)
            except Exception:
                _mois = ["janvier","février","mars","avril","mai","juin",
                        "juillet","août","septembre","octobre","novembre","décembre"]
                m = pd.to_datetime(df_display["month"])
                mois_labels = m.dt.month.map(lambda i: _mois[i-1]).str.capitalize() + " " + m.dt.year.astype(str)

            df_display["Mois"] = mois_labels
            df_display = df_display.rename(columns={
                "pct_brut": "Disponibilité brute",
                "pct_excl": "Avec exclusions",
                "total_minutes": "Durée totale"
            })[["Mois", "Disponibilité brute", "Avec exclusions", "Durée totale"]]

            def _fmt_duree(x):
                try:
                    return format_minutes(int(x))  
                except Exception:
                    return "—"

            st.dataframe(
                df_display.style.format({
                    "Disponibilité brute": "{:.2f}%",
                    "Avec exclusions": "{:.2f}%",
                    "Durée totale": _fmt_duree
                }),
                hide_index=True,
                use_container_width=True
            )


def render_global_comparison_tab(start_dt: datetime, end_dt: datetime) -> None:
    """Affiche la vue comparative entre tous les sites."""
    mode = get_current_mode()
    st.header("🌍 Vue générale - Comparaison tous les sites")
    st.caption(
        f"Période analysée : {start_dt.strftime('%Y-%m-%d')} ➜ {end_dt.strftime('%Y-%m-%d')}"
    )

    df_all = load_filtered_blocks(start_dt, end_dt, None, None, mode=mode)

    if df_all is None or df_all.empty:
        st.warning("Aucune donnée disponible pour la vue globale sur la période sélectionnée.")
        return

    if mode == MODE_EQUIPMENT:
        st.subheader("Récap AC / DC1 / DC2")
        if "type_equipement" not in df_all.columns:
            st.info("Les données de type équipement ne sont pas disponibles pour cette vue.")
            return

        base_types = ["AC", "DC1", "DC2"]
        additional_types = [
            t for t in df_all["type_equipement"].dropna().unique().tolist()
            if t not in base_types
        ]
        type_sequence = base_types + additional_types

        site_rows: List[Dict[str, Optional[float]]] = []
        for site, site_df in df_all.groupby("site"):
            site_label = mapping_sites.get(str(site).split("_")[-1], str(site))
            row: Dict[str, Optional[float]] = {"Site": site_label}
            for equip_type in type_sequence:
                type_df = site_df[site_df["type_equipement"] == equip_type]
                column_label = f"{equip_type} (%)"
                if type_df.empty:
                    row[column_label] = math.nan
                else:
                    stats = calculate_availability(type_df, include_exclusions=False)
                    row[column_label] = round(stats["pct_available"], 2)
            site_rows.append(row)

        summary_df = pd.DataFrame(site_rows)
        summary_df = summary_df.dropna(axis=1, how="all")
        if summary_df.empty:
            st.info("Aucune donnée consolidée disponible pour les sites.")
        else:
            summary_df = summary_df.sort_values("Site").reset_index(drop=True)
            column_config = {
                "Site": st.column_config.TextColumn("Site", width="medium"),
            }
            for column in summary_df.columns:
                if column == "Site":
                    continue
                column_config[column] = st.column_config.NumberColumn(
                    column,
                    width="small",
                    format="%.2f%%"
                )

            st.dataframe(
                summary_df,
                hide_index=True,
                use_container_width=True,
                column_config=column_config,
            )

        present_types = [
            t for t in type_sequence
            if not df_all[df_all["type_equipement"] == t].empty
        ]
        if present_types:
            cols = st.columns(len(present_types))
            for col, equip_type in zip(cols, present_types):
                type_df = df_all[df_all["type_equipement"] == equip_type]
                stats_raw = calculate_availability(type_df, include_exclusions=False)
                stats_excl = calculate_availability(type_df, include_exclusions=True)
                delta = stats_excl["pct_available"] - stats_raw["pct_available"]
                col.metric(
                    f"{equip_type} - disponibilité brute",
                    f"{stats_raw['pct_available']:.2f}%",
                    delta=f"{delta:.2f}%",
                    help="Comparaison agrégée sur l'ensemble des sites",
                )
    else:
        st.subheader("Dispo globale par site")
        site_rows = []
        for site, site_df in df_all.groupby("site"):
            stats_raw = calculate_availability(site_df, include_exclusions=False)
            stats_excl = calculate_availability(site_df, include_exclusions=True)
            site_rows.append({
                "Site": mapping_sites.get(str(site).split("_")[-1], str(site)),
                "Disponibilité brute (%)": round(stats_raw["pct_available"], 2),
                "Disponibilité avec exclusions (%)": round(stats_excl["pct_available"], 2),
            })

        summary_df = pd.DataFrame(site_rows)
        if summary_df.empty:
            st.info("Aucune donnée consolidée disponible pour les sites.")
        else:
            summary_df = summary_df.sort_values("Site").reset_index(drop=True)
            st.dataframe(
                summary_df,
                hide_index=True,
                use_container_width=True,
                column_config={
                    "Site": st.column_config.TextColumn("Site", width="medium"),
                    "Disponibilité brute (%)": st.column_config.NumberColumn(
                        "Disponibilité brute (%)",
                        width="medium",
                        format="%.2f%%",
                    ),
                    "Disponibilité avec exclusions (%)": st.column_config.NumberColumn(
                        "Disponibilité avec exclusions (%)",
                        width="medium",
                        format="%.2f%%",
                    ),
                },
            )

        stats_all_raw = calculate_availability(df_all, include_exclusions=False)
        stats_all_excl = calculate_availability(df_all, include_exclusions=True)
        delta = stats_all_excl["pct_available"] - stats_all_raw["pct_available"]
        col1, col2 = st.columns(2)
        col1.metric(
            "Disponibilité brute globale",
            f"{stats_all_raw['pct_available']:.2f}%",
            help="Disponibilité brute de l'ensemble des points de charge",
        )
        col2.metric(
            "Disponibilité avec exclusions globale",
            f"{stats_all_excl['pct_available']:.2f}%",
            delta=f"{delta:.2f}%",
            help="Comparaison brute vs exclusions sur tous les sites",
        )


def render_timeline_tab(site: Optional[str], equip: Optional[str], start_dt: datetime, end_dt: datetime):
    """Affiche l'onglet timeline et annotations."""
    mode = get_current_mode()
    st.header("⏱️ Timeline Détaillée & Annotations")
    
    if not site or not equip:
        st.info("ℹ️ Veuillez sélectionner un site et un équipement spécifiques pour afficher la timeline détaillée.")
        return
    
    with st.spinner("Chargement de la timeline..."):
        df = load_blocks(site, equip, start_dt, end_dt, mode=mode)
    
    if df.empty:
        st.warning("⚠️ Aucune donnée disponible pour cet équipement sur cette période.")
        return
    
    df_plot = df.copy()
    df_plot["start"] = pd.to_datetime(df_plot["date_debut"])
    df_plot["end"] = pd.to_datetime(df_plot["date_fin"])
    df_plot["state"] = df_plot["est_disponible"].map({
        1: "✅ Disponible",
        0: "❌ Indisponible",
        -1: "⚠️ Donnée manquante"
    })

    df_plot["excluded"] = ""
    mask_excluded = df_plot["is_excluded"] == 1
    df_plot.loc[mask_excluded, "excluded"] = " (Exclu)"
    df_plot["label"] = df_plot["state"] + df_plot["excluded"]
    
    fig = px.timeline(
        df_plot,
        x_start="start",
        x_end="end",
        y="equipement_id",
        color="label",
        hover_data={
            "cause": True,
            "duration_minutes": True,
            "is_excluded": True,
            "start": "|%Y-%m-%d %H:%M",
            "end": "|%Y-%m-%d %H:%M",
            "label": False,
            "equipement_id": False
        },
        color_discrete_map={
            "✅ Disponible": "#28a745",
            "✅ Disponible (Exclu)": "#17a2b8",
            "❌ Indisponible": "#dc3545",
            "❌ Indisponible (Exclu)": "#fd7e14",
            "⚠️ Donnée manquante": "#6c757d",
            "⚠️ Donnée manquante (Exclu)": "#BBDB07"
        }
    )
    
    fig.update_yaxes(autorange="reversed", title="")
    fig.update_xaxes(title="Période")
    fig.update_layout(
        title=f"Timeline - {site} / {equip}",
        showlegend=True,
        height=300
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Tableau des vraies périodes d'indisponibilité (groupées)
    st.markdown("**📋 Périodes d'Indisponibilité Continues :**")
    
    # Filtrer les indisponibilités
    unavailable_periods = df[df["est_disponible"] == 0].copy()
    
    if not unavailable_periods.empty:
        # Trier par date de début
        unavailable_periods = unavailable_periods.sort_values("date_debut").reset_index(drop=True)
        
        # Grouper les périodes continues
        continuous_periods = []
        current_period = None
        
        for _, row in unavailable_periods.iterrows():
            start_time = pd.to_datetime(row["date_debut"])
            end_time = pd.to_datetime(row["date_fin"])
            
            if current_period is None:
                # Première période
                current_period = {
                    "start": start_time,
                    "end": end_time,
                    "causes": [row["cause"] if pd.notna(row["cause"]) else "Non spécifiée"],
                    "excluded": row["is_excluded"] == 1,
                    "duration_minutes": int(row["duration_minutes"])
                }
            else:
                # Vérifier si cette période est continue avec la précédente
                # (écart de moins de 5 minutes considéré comme continu)
                gap_minutes = (start_time - current_period["end"]).total_seconds() / 60
                
                if gap_minutes <= 5:  # Période continue
                    # Étendre la période actuelle
                    current_period["end"] = end_time
                    current_period["causes"].append(row["cause"] if pd.notna(row["cause"]) else "Non spécifiée")
                    current_period["duration_minutes"] += int(row["duration_minutes"])
                    # Si une période est exclue, toute la période continue est considérée comme exclue
                    if row["is_excluded"] == 1:
                        current_period["excluded"] = True
                else:
                    # Nouvelle période - sauvegarder la précédente
                    continuous_periods.append(current_period)
                    # Commencer une nouvelle période
                    current_period = {
                        "start": start_time,
                        "end": end_time,
                        "causes": [row["cause"] if pd.notna(row["cause"]) else "Non spécifiée"],
                        "excluded": row["is_excluded"] == 1,
                        "duration_minutes": int(row["duration_minutes"])
                    }
        
        # Ajouter la dernière période
        if current_period is not None:
            continuous_periods.append(current_period)
        
        if continuous_periods:
            # Préparer les données pour le tableau
            periods_data = []
            for i, period in enumerate(continuous_periods, 1):
                # Calculer la durée totale de la période continue
                total_duration_minutes = int((period["end"] - period["start"]).total_seconds() / 60)
                
                # Créer un résumé des causes (prendre les causes uniques)
                unique_causes = list(set(period["causes"]))
                if len(unique_causes) == 1:
                    cause_summary = unique_causes[0]
                else:
                    cause_summary = f"{len(unique_causes)} causes différentes"
                
                periods_data.append({
                    "Période": f"Période {i}",
                    "Date Début": period["start"].strftime("%Y-%m-%d %H:%M"),
                    "Date Fin": period["end"].strftime("%Y-%m-%d %H:%M"),
                    "Durée": format_minutes(total_duration_minutes),
                    "Durée_Minutes": total_duration_minutes,
                    "Cause": cause_summary,
                    "Exclu": "✅ Oui" if period["excluded"] else "❌ Non"
                })
            
            # Créer le DataFrame et trier par durée décroissante
            periods_df = pd.DataFrame(periods_data)
            periods_sorted = periods_df.sort_values("Durée_Minutes", ascending=False)
            
            st.dataframe(
                periods_sorted[["Période", "Date Début", "Date Fin", "Durée", "Cause", "Exclu"]],
                hide_index=True,
                use_container_width=True,
                column_config={
                    "Période": st.column_config.TextColumn("Période", width="small"),
                    "Date Début": st.column_config.TextColumn("Date Début", width="medium"),
                    "Date Fin": st.column_config.TextColumn("Date Fin", width="medium"),
                    "Durée": st.column_config.TextColumn("Durée", width="medium"),
                    "Cause": st.column_config.TextColumn("Cause", width="large"),
                    "Exclu": st.column_config.TextColumn("Exclu", width="small")
                }
            )
            
            # Métriques rapides
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Périodes", len(periods_data))
            with col2:
                total_duration = periods_df["Durée_Minutes"].sum()
                st.metric("Durée Totale", format_minutes(total_duration))
            with col3:
                avg_duration = periods_df["Durée_Minutes"].mean()
                st.metric("Durée Moyenne", format_minutes(int(avg_duration)))
            with col4:
                max_duration = periods_df["Durée_Minutes"].max()
                st.metric("Durée Max", format_minutes(max_duration))
        else:
            st.success("✅ Aucune période d'indisponibilité continue détectée.")
    else:
        st.success("✅ Aucune période d'indisponibilité détectée sur cette période.")
    
    st.divider()
    st.subheader("➕ Ajouter une Annotation")

    mode = st.radio(
        "Afficher",
        options=["Disponibles", "Indisponibles", "Données manquantes"],
        index=1,           
        horizontal=True
    )

    if mode == "Disponibles":
        df_display = df_plot[df_plot["est_disponible"] == 1]
    elif mode == "Indisponibles":
        df_display = df_plot[df_plot["est_disponible"] == 0]
    else:  
        df_display = df_plot[df_plot["est_disponible"] == -1]

    if df_display.empty:
        st.info("ℹ️ Aucun bloc correspondant aux critères d'affichage.")
    else:
        df_display = df_display.sort_values("start").reset_index(drop=True)

        block_labels = []
        for idx, row in df_display.iterrows():
            if row["est_disponible"] == -1:
                status_icon = "⚠️"
            elif row["est_disponible"] == 0:
                status_icon = "❌"
            else:
                status_icon = "✅"

            excl_tag = " [EXCLU]" if row["is_excluded"] == 1 else ""
            start_str = row["start"].strftime("%Y-%m-%d %H:%M")
            end_str = row["end"].strftime("%Y-%m-%d %H:%M")
            cause = row.get("cause", "N/A")
            duration = format_minutes(int(row["duration_minutes"]))

            label = f"{idx}: {status_icon} {start_str} → {end_str} | {cause} | {duration}{excl_tag}"
            block_labels.append(label)

        selected_block_label = st.selectbox(
            "Sélectionner un bloc temporel",
            options=block_labels,
            help="Choisissez le bloc sur lequel ajouter une annotation"
        )

        selected_idx = int(selected_block_label.split(":")[0])
        selected_row = df_display.iloc[selected_idx]
        est_val = int(selected_row["est_disponible"])

        bloc_id = int(selected_row.get("bloc_id", -1))
        source_table = str(selected_row.get("source_table", "") or "")

        active_exclusion = bool(int(selected_row.get("is_excluded", 0)))
        exclusion_id = selected_row.get("exclusion_id")

        st.markdown("### 🚫 Gestion de l'exclusion du bloc")
        if bloc_id <= 0 or not source_table:
            st.warning(
                "⚠️ Impossible d'identifier ce bloc dans la base : aucune action d'exclusion n'est possible."
            )
        else:
            if active_exclusion:
                st.info("Ce bloc est actuellement exclu des calculs.")
                applied_by = selected_row.get("exclusion_applied_by")
                applied_at = selected_row.get("exclusion_applied_at")
                applied_comment = selected_row.get("exclusion_comment")
                previous_status = int(selected_row.get("previous_status", est_val))

                with st.expander("Détails de l'exclusion active", expanded=True):
                    st.write(
                        {
                            "Exclusion #": exclusion_id or "—",
                            "Appliquée par": applied_by or "—",
                            "Appliquée le": applied_at.strftime("%Y-%m-%d %H:%M") if isinstance(applied_at, datetime) else str(applied_at or "—"),
                            "Statut initial": {1: "Disponible", 0: "Indisponible", -1: "Donnée manquante"}.get(previous_status, "Inconnu"),
                            "Commentaire": applied_comment or "—",
                        }
                    )

                with st.form(f"release_exclusion_{bloc_id}"):
                    release_operator = st.text_input(
                        "Opérateur (historisation)",
                        value="",
                        placeholder="ex: Jean Dupont",
                        help="Identifiez la personne qui supprime l'exclusion.",
                    )
                    release_comment = st.text_area(
                        "Commentaire de réactivation",
                        placeholder="Décrivez pourquoi cette exclusion est levée...",
                        help="Un commentaire détaillé est requis pour tracer le rollback.",
                    )
                    submit_release = st.form_submit_button("♻️ Lever l'exclusion et restaurer l'état d'origine")

                    if submit_release:
                        release_txt = release_comment.strip()
                        if len(release_txt) < 5:
                            st.error("❌ Le commentaire doit contenir au moins 5 caractères.")
                        else:
                            try:
                                result = release_block_exclusion(
                                    table_name=source_table,
                                    block_id=bloc_id,
                                    user=release_operator.strip() or None,
                                    comment=release_txt,
                                )
                            except ExclusionError as exc:
                                st.error(f"❌ Impossible de lever l'exclusion : {exc}")
                            else:
                                st.success(
                                    f"✅ Bloc {result.block_id} restauré avec le statut {result.new_status} (table {result.table_name})."
                                )
                                st.balloons()
                                st.rerun()
            else:
                st.warning("Ce bloc est actuellement comptabilisé normalement.")
                with st.form(f"apply_exclusion_{bloc_id}"):
                    exclusion_operator = st.text_input(
                        "Opérateur (historisation)",
                        value="",
                        placeholder="ex: Jean Dupont",
                        help="Identifiez la personne à l'origine de l'exclusion.",
                    )
                    exclusion_comment = st.text_area(
                        "Commentaire obligatoire",
                        placeholder="Décrivez pourquoi cette période doit être exclue...",
                        help="Ce commentaire sera stocké pour permettre un rollback.",
                    )
                    target_status = 1
                    if est_val == -1:
                        status_labels = {
                            1: "✅ Disponible (1)",
                            0: "⛔ Indisponible (0)",
                        }
                        target_status = st.radio(
                            "Statut à appliquer pendant l'exclusion",
                            options=[1, 0],
                            index=0,
                            format_func=lambda x: status_labels.get(int(x), str(x)),
                            horizontal=True,
                            help="Choisissez le statut à appliquer lorsque les données sont manquantes.",
                        )
                    else:
                        st.caption(
                            "Le bloc sera marqué comme disponible pendant la période d'exclusion."
                        )

                    button_label = (
                        "🚫 Exclure ce bloc et le marquer comme "
                        f"{'disponible' if int(target_status) == 1 else 'indisponible'}"
                    )
                    submit_exclusion = st.form_submit_button(button_label)

                    if submit_exclusion:
                        comment_txt = exclusion_comment.strip()
                        if len(comment_txt) < 5:
                            st.error("❌ Le commentaire doit contenir au moins 5 caractères.")
                        else:
                            try:
                                result = apply_block_exclusion(
                                    table_name=source_table,
                                    block_id=bloc_id,
                                    user=exclusion_operator.strip() or None,
                                    comment=comment_txt,
                                    new_status=int(target_status),
                                )
                            except ExclusionError as exc:
                                st.error(f"❌ Impossible de créer l'exclusion : {exc}")
                            else:
                                st.success(
                                    "✅ Bloc {bloc} exclu et marqué {statut} (table {table}).".format(
                                        bloc=result.block_id,
                                        statut="disponible" if result.new_status == 1 else "indisponible",
                                        table=result.table_name,
                                    )
                                )
                                st.balloons()
                                st.rerun()

            st.markdown(f"**Bloc sélectionné:** {selected_row['start']} → {selected_row['end']}")
            if est_val != 1: 
                cause_originale = selected_row.get("cause", "Non spécifié")
                equip_current = st.session_state.get("current_equip")
                
                if equip_current and cause_originale != "Non spécifié":
                    cause_traduite = translate_cause_to_text(cause_originale, equip_current)
                    
                    if cause_traduite != cause_originale:
                        st.markdown("**🔧 Cause d'indisponibilité traduite :**")
                        st.info(f"**Original :** {cause_originale}\n\n**Traduit :** {cause_traduite}")
                    else:
                        st.markdown("**🔧 Cause d'indisponibilité :**")
                        st.info(f"**Cause :** {cause_originale}")
                else:
                    st.markdown("**🔧 Cause d'indisponibilité :**")
                    st.info(f"**Cause :** {cause_originale}")
                    st.rerun()

        st.markdown("---")  
        st.markdown("### 💬 Ajouter un Commentaire")
        st.caption("Ajouter un commentaire informatif pour ce bloc sans affecter les calculs de disponibilité.")

        with st.form(f"add_comment_{bloc_id}"):
            comment_operator = st.text_input(
                "Créé par",
                value="",
                placeholder="ex: Jean Dupont",
                help="Identifiez la personne qui ajoute ce commentaire.",
            )
            
            comment_text = st.text_area(
                "Commentaire",
                placeholder="Décrivez l'événement, l'observation ou toute information pertinente pour cette période...",
                help="Ce commentaire sera visible dans l'onglet 'Gestion des Commentaires'.",
                height=120
            )
            
            submit_comment = st.form_submit_button("💬 Ajouter le commentaire", type="primary")
            
            if submit_comment:
                comment_txt = comment_text.strip()
                if len(comment_txt) < 5:
                    st.error("❌ Le commentaire doit contenir au moins 5 caractères.")
                else:
                    try:
                        # Récupérez le site et l'équipement depuis la session
                        site = st.session_state.get("current_site", "")
                        equipement = st.session_state.get("current_equip", "")
                        
                        # Utilisez directement les dates du bloc sélectionné
                        success = create_annotation(
                            site=site,
                            equip=equipement,
                            start_dt=selected_row["start"],
                            end_dt=selected_row["end"],
                            annotation_type="commentaire",
                            comment=comment_txt,
                            user=comment_operator.strip() or "ui",
                            cascade=False
                        )
                        
                        if success:
                            st.success(f"✅ Commentaire ajouté avec succès !")
                            st.balloons()
                            st.rerun()
                        else:
                            st.error("❌ Échec de l'ajout du commentaire.")
                        
                    except Exception as exc:
                        st.error(f"❌ Impossible d'ajouter le commentaire : {exc}")
    with st.expander("⚡ Exclusion rapide des données manquantes", expanded=False):
        month_default = datetime.utcnow().date().replace(day=1)
        month_candidates = [
            ts.to_pydatetime().date() for ts in pd.date_range(end=month_default, periods=12, freq="MS")
        ]
        month_candidates.reverse()
        default_index = month_candidates.index(month_default) if month_default in month_candidates else 0
        target_month = st.selectbox(
            "Mois concerné",
            options=month_candidates,
            index=default_index,
            format_func=lambda d: d.strftime("%Y-%m"),
            key="timeline_missing_month_picker",
            help="Choisissez un mois pour exclure automatiquement toutes les données manquantes.",
        )

        month_start = target_month.replace(day=1)
        if month_start.month == 12:
            next_month = month_start.replace(year=month_start.year + 1, month=1)
        else:
            next_month = month_start.replace(month=month_start.month + 1)

        st.markdown("**Sites concernés par l'exclusion automatique**")
        exclusion_sites = ["AC", "DC1", "DC2", "PDC1", "PDC2", "PDC3", "PDC4", "PDC5", "PDC6"]
        site_columns = st.columns(3)
        selected_sites = []
        for idx, site_label in enumerate(exclusion_sites):
            col = site_columns[idx % len(site_columns)]
            if col.checkbox(site_label, key=f"timeline_missing_site_{site_label.lower()}"):
                selected_sites.append(site_label)

        st.session_state["timeline_missing_selected_sites"] = selected_sites

        default_comment = f"Exclusion automatique données manquantes {month_start.strftime('%Y-%m')}"
        bulk_comment = st.text_input(
            "Commentaire appliqué",
            value=default_comment,
            key="timeline_missing_month_comment",
            help="Le commentaire sera répliqué sur chaque exclusion créée.",
        )
        bulk_user = st.text_input(
            "Créé par",
            placeholder="Votre nom",
            key="timeline_missing_month_user",
        )

        col_apply_available, col_apply_unavailable = st.columns(2)
        trigger_available = col_apply_available.button(
            "✅ Exclure comme disponible",
            key="timeline_missing_exclude_available",
            use_container_width=True,
        )
        trigger_unavailable = col_apply_unavailable.button(
            "❌ Exclure comme indisponible",
            key="timeline_missing_exclude_unavailable",
            use_container_width=True,
        )

        if trigger_available or trigger_unavailable:
            if not selected_sites:
                st.warning("Sélectionnez au moins un équipement à traiter.")
            else:
                site_scope = st.session_state.get("current_site")
                if not site_scope:
                    st.error(
                        "Sélectionnez un site spécifique dans les filtres avant d'utiliser l'exclusion automatique."
                    )
                else:
                    comment_txt = (bulk_comment or "").strip()
                    if len(comment_txt) < 5:
                        st.error("Le commentaire doit contenir au moins 5 caractères.")
                    else:
                        user_txt = (bulk_user or "").strip()
                        start_dt = datetime.combine(month_start, time.min)
                        end_dt = datetime.combine(next_month, time.min)
                        new_status = 1 if trigger_available else 0
                        status_label = "disponible" if new_status == 1 else "indisponible"

                        available_equips = {s.upper() for s in get_equipments(MODE_EQUIPMENT, site_scope) or []}
                        available_pdc = {s.upper() for s in get_equipments(MODE_PDC, site_scope) or []}

                        total_created = 0
                        total_candidates = 0
                        info_messages: List[str] = []
                        error_messages: List[str] = []

                        with st.spinner("Application des exclusions automatiques..."):
                            for equip_label in selected_sites:
                                equip_upper = equip_label.upper()
                                mode = MODE_PDC if equip_upper.startswith("PDC") else MODE_EQUIPMENT

                                if mode == MODE_PDC and equip_upper not in available_pdc:
                                    info_messages.append(
                                        f"{equip_label}: aucun point de charge correspondant pour le site sélectionné."
                                    )
                                    continue

                                if mode == MODE_EQUIPMENT and equip_upper not in available_equips:
                                    info_messages.append(
                                        f"{equip_label}: équipement indisponible sur le site sélectionné."
                                    )
                                    continue

                                created, candidates, errors = _bulk_exclude_missing_blocks(
                                    site=site_scope,
                                    equip=equip_label,
                                    start_dt=start_dt,
                                    end_dt=end_dt,
                                    new_status=new_status,
                                    comment=comment_txt,
                                    user=user_txt or None,
                                )

                                total_created += created
                                total_candidates += candidates
                                error_messages.extend(errors)

                                if candidates == 0:
                                    info_messages.append(
                                        f"{equip_label}: aucune donnée manquante sur la période sélectionnée."
                                    )

                        if total_created > 0:
                            st.success(
                                f"✅ {total_created} exclusion{'s' if total_created > 1 else ''} créée{'s' if total_created > 1 else ''}"
                                f" et marquée{'s' if total_created > 1 else ''} comme {status_label}."
                            )

                            if info_messages:
                                st.info("\n".join(info_messages))

                            if error_messages:
                                st.warning("\n".join(error_messages))

                            st.balloons()
                            st.rerun()
                        else:
                            if info_messages:
                                st.info("\n".join(info_messages))
                            if error_messages:
                                st.error("\n".join(error_messages))
                            if not info_messages and not error_messages:
                                st.info(
                                    "Aucune donnée manquante à exclure pour la période et la sélection indiquées."
                                )
def render_exclusions_tab():
    mode = get_current_mode()
    st.header("🚫 Gestion des exclusions")

    st.markdown(
        """
        Les exclusions actives sont appliquées directement sur les blocs de disponibilité.
        Utilisez la timeline pour créer de nouvelles exclusions et ce panneau pour consulter
        ou lever celles qui sont encore actives.
        """
    )

    st.subheader("🔒 Exclusions actives")
    df_active = get_block_exclusions(active_only=True, limit=200)
    if df_active.empty:
        st.success("✅ Aucune exclusion active dans la base de données.")
    else:
        status_map = {1: "Disponible", 0: "Indisponible", -1: "Donnée manquante"}

        def _format_applied_at(value: Any) -> str:
            timestamp = pd.to_datetime(value, errors="coerce")
            if pd.isna(timestamp):
                return "—"
            return timestamp.strftime("%Y-%m-%d %H:%M")

        def _format_duration(delta: Optional[pd.Timedelta]) -> str:
            if delta is None or pd.isna(delta):
                return "—"
            total_minutes = max(int(delta.total_seconds() // 60), 0)
            days, remainder = divmod(total_minutes, 1440)
            hours, minutes = divmod(remainder, 60)
            parts: List[str] = []
            if days:
                parts.append(f"{days} j")
            if hours:
                parts.append(f"{hours} h")
            if minutes or not parts:
                parts.append(f"{minutes} min")
            return " ".join(parts)

        now_ts = pd.Timestamp.utcnow()
        applied_ts = pd.to_datetime(df_active["applied_at"], errors="coerce")

        df_active_display = pd.DataFrame(
            {
                "Sélection": [False] * len(df_active),
                "ID exclusion": df_active["id"].astype(int),
                "Table": df_active["table_name"],
                "Bloc": df_active["bloc_id"].astype(int),
                "Statut initial": df_active["previous_status"].map(status_map).fillna("Inconnu"),
                "Commentaire": df_active["exclusion_comment"].fillna("—"),
                "Appliquée par": df_active["applied_by"].fillna("—"),
                "Appliquée le": applied_ts.map(_format_applied_at),
                "Actif depuis": (now_ts - applied_ts).map(_format_duration),
            }
        )

        st.caption("Sélectionnez une ou plusieurs exclusions pour afficher les détails et les lever en lot.")
        edited_active = st.data_editor(
            df_active_display,
            hide_index=True,
            num_rows="fixed",
            column_config={
                "Sélection": st.column_config.CheckboxColumn(
                    "Sélection",
                    help="Cochez pour inclure l'exclusion dans la sélection courante.",
                    default=False,
                ),
                "Commentaire": st.column_config.TextColumn(disabled=True),
                "Appliquée par": st.column_config.TextColumn(disabled=True),
                "Appliquée le": st.column_config.TextColumn(disabled=True),
                "Actif depuis": st.column_config.TextColumn(disabled=True),
            },
            disabled=["ID exclusion", "Table", "Bloc", "Statut initial"],
            key="active_exclusions_editor",
        )

        selected_ids = [
            int(row["ID exclusion"])
            for _, row in edited_active.iterrows()
            if bool(row.get("Sélection"))
        ]

        if selected_ids:
            st.info(
                f"{len(selected_ids)} exclusion{'s' if len(selected_ids) > 1 else ''} sélectionnée{'s' if len(selected_ids) > 1 else ''}."
            )

            selected_details = df_active[df_active["id"].isin(selected_ids)]
            with st.expander("Détails des exclusions sélectionnées", expanded=False):
                for _, selected in selected_details.sort_values("applied_at", ascending=False).iterrows():
                    st.markdown(
                        f"**Bloc #{int(selected['bloc_id'])} · {selected['table_name']}**"
                    )
                    st.write(
                        {
                            "ID": int(selected["id"]),
                            "Statut initial": status_map.get(int(selected.get("previous_status", -1)), "Inconnu"),
                            "Commentaire": selected.get("exclusion_comment") or "—",
                            "Appliquée par": selected.get("applied_by") or "—",
                            "Appliquée le": _format_applied_at(selected.get("applied_at")),
                        }
                    )

        with st.form("bulk_release_form"):
            release_operator = st.text_input(
                "Opérateur (historisation)",
                placeholder="ex: Jean Dupont",
            )
            release_comment = st.text_area(
                "Commentaire de réactivation",
                placeholder="Expliquez pourquoi les exclusions sélectionnées sont levées",
            )
            submit_bulk_release = st.form_submit_button("♻️ Lever les exclusions sélectionnées", disabled=not selected_ids)

        if submit_bulk_release:
            comment_txt = release_comment.strip()
            if not selected_ids:
                st.warning("Veuillez sélectionner au moins une exclusion active à lever.")
            elif len(comment_txt) < 5:
                st.error("❌ Le commentaire doit contenir au moins 5 caractères.")
            else:
                successes = 0
                errors: List[str] = []
                for exclusion_id in selected_ids:
                    row = df_active[df_active["id"] == exclusion_id].iloc[0]
                    try:
                        release_block_exclusion(
                            table_name=str(row["table_name"]),
                            block_id=int(row["bloc_id"]),
                            user=release_operator.strip() or None,
                            comment=comment_txt,
                        )
                    except ExclusionError as exc:
                        errors.append(f"Exclusion #{exclusion_id}: {exc}")
                    else:
                        successes += 1

                if successes:
                    st.success(
                        f"✅ {successes} exclusion{'s' if successes > 1 else ''} levée{'s' if successes > 1 else ''} avec succès."
                    )
                if errors:
                    st.error("\n".join(f"❌ {message}" for message in errors))

                if successes:
                    st.rerun()
    st.subheader("🕒 Historique récent")
    df_history = get_block_exclusions(active_only=False, limit=200)
    if df_history.empty:
        st.info("ℹ️ Aucun historique disponible.")
    else:
        history = df_history.copy()
        history["Statut"] = history["released_at"].apply(lambda v: "✅ Active" if pd.isna(v) else "❌ Levée")
        status_map = {1: "Disponible", 0: "Indisponible", -1: "Donnée manquante"}
        history["Statut initial"] = history["previous_status"].map(status_map).fillna("Inconnu")
        history["Appliquée le"] = pd.to_datetime(history["applied_at"]).dt.strftime("%Y-%m-%d %H:%M")
        history["Levée le"] = pd.to_datetime(history["released_at"]).dt.strftime("%Y-%m-%d %H:%M")
        display_cols = [
            "id",
            "table_name",
            "bloc_id",
            "Statut",
            "Statut initial",
            "exclusion_comment",
            "applied_by",
            "Appliquée le",
            "released_by",
            "Levée le",
            "release_comment",
        ]
        history = history[display_cols]
        st.dataframe(
            history,
            hide_index=True,
            use_container_width=True,
            column_config={
                "id": st.column_config.NumberColumn("ID", width="small"),
                "table_name": st.column_config.TextColumn("Table", width="medium"),
                "bloc_id": st.column_config.NumberColumn("Bloc", width="small"),
                "Statut initial": st.column_config.TextColumn("Statut initial", width="medium"),
                "exclusion_comment": st.column_config.TextColumn("Commentaire", width="large"),
                "applied_by": st.column_config.TextColumn("Appliquée par", width="medium"),
                "released_by": st.column_config.TextColumn("Levée par", width="medium"),
                "release_comment": st.column_config.TextColumn("Commentaire de levée", width="large"),
            },
        )

def render_comments_tab():
    """Affiche l'onglet de gestion des commentaires."""
    st.header("💬 Gestion des Commentaires")
    
    st.markdown("""
    Les **commentaires** sont des annotations informatives qui n'affectent pas 
    le calcul de disponibilité mais permettent de documenter des événements ou observations.
    """)
    
    st.divider()
    
    st.subheader("📋 Commentaires Existants")
    
    df_comments = get_annotations(annotation_type="commentaire", limit=200)
    
    if df_comments.empty:
        st.info("ℹ️ Aucun commentaire enregistré pour le moment.")
    else:
        df_display = df_comments.copy()
        df_display["Période"] = df_display.apply(
            lambda r: f"{pd.to_datetime(r['date_debut']).strftime('%Y-%m-%d %H:%M')} → {pd.to_datetime(r['date_fin']).strftime('%Y-%m-%d %H:%M')}",
            axis=1
        )
        df_display["Créé le"] = pd.to_datetime(df_display["created_at"]).dt.strftime("%Y-%m-%d %H:%M")
        df_display["Statut"] = df_display["actif"].map({1: "✅ Actif", 0: "❌ Inactif"})
        
        columns_config = [
            ("id", "ID", 0.8),
            ("site", "Site", 1.1),
            ("equipement_id", "Équipement", 1.2),
            ("Période", "Période", 1.8),
            ("commentaire", "Commentaire", 2.5),
            ("Statut", "Statut", 1.0),
            ("created_by", "Créé par", 1.2),
            ("Créé le", "Créé le", 1.3),
        ]

        st.caption("Cliquez sur 🗑️ pour supprimer un commentaire directement depuis la liste.")
        render_inline_delete_table(
            df_display,
            column_settings=columns_config,
            key_prefix="comment",
            delete_handler=delete_annotation,
            success_message="✅ Commentaire #{id} supprimé !",
            error_message="❌ Échec de suppression pour le commentaire #{id}."
        )
        
        st.subheader("✏️ Éditer un Commentaire")
        selected_id = st.number_input(
            "ID du commentaire à éditer",
            min_value=0,
            value=0,
            step=1,
            help="Entrez l'ID du commentaire à modifier"
        )
        
        if selected_id > 0:
            selected_comment = df_comments[df_comments["id"] == selected_id]
            
            if selected_comment.empty:
                st.error(f"❌ Aucun commentaire trouvé avec l'ID {selected_id}")
            else:
                comment_info = selected_comment.iloc[0]
                current_text = comment_info["commentaire"]
                
                st.info(f"""
                **Commentaire #{selected_id}**  
                📍 Site: {comment_info['site']} | Équipement: {comment_info['equipement_id']}  
                📅 Période: {pd.to_datetime(comment_info['date_debut']).strftime('%Y-%m-%d %H:%M')} → {pd.to_datetime(comment_info['date_fin']).strftime('%Y-%m-%d %H:%M')}  
                👤 Créé par: {comment_info['created_by']}
                """)
                
                new_text = st.text_area(
                    "Nouveau texte du commentaire",
                    value=current_text,
                    height=150,
                    help="Modifiez le texte du commentaire"
                )
                
                col1, col2 = st.columns(2)

                with col1:
                    if st.button("💾 Enregistrer les modifications", type="primary", use_container_width=True):
                        if not new_text :
                            st.error("❌ Veuillez mettre un commentaire.")
                        else:
                            if update_annotation_comment(selected_id, new_text.strip()):
                                st.success(f"✅ Commentaire #{selected_id} mis à jour !")
                                st.rerun()
                
                with col2:
                    is_active = comment_info["actif"] == 1
                    if is_active:
                        if st.button("❌ Désactiver", use_container_width=True):
                            if toggle_annotation(selected_id, False):
                                st.warning(f"⚠️ Commentaire #{selected_id} désactivé !")
                                st.rerun()
                    else:
                        if st.button("✅ Activer", use_container_width=True):
                            if toggle_annotation(selected_id, True):
                                st.success(f"✅ Commentaire #{selected_id} activé !")
                                st.rerun()

                st.caption("🗑️ Utilisez la liste ci-dessus pour supprimer un commentaire.")



@dataclass
class EquipmentReportDetail:
    """Structure contenant les données préparées pour l'affichage du rapport."""

    name: str
    summary: Optional[Dict[str, str]]
    unavailable_table: pd.DataFrame
    missing_table: pd.DataFrame
    causes_table: pd.DataFrame
    daily_table: pd.DataFrame
    unavailable_minutes: int = 0
    missing_minutes: int = 0
    excluded_events: int = 0


def _prepare_report_summary(
    report_data: Dict[str, pd.DataFrame],
    equipments: List[str],
) -> Tuple[pd.DataFrame, Dict[str, EquipmentReportDetail], Dict[str, float]]:
    """Construit les différentes vues utilisées dans l'onglet rapport."""

    overview_rows: List[Dict[str, object]] = []
    equipment_details: Dict[str, EquipmentReportDetail] = {}

    total_unavailable_minutes = 0
    total_missing_minutes = 0
    total_unavailable_events = 0
    total_missing_events = 0
    total_exclusions = 0
    availability_values: List[float] = []

    jours_fr = {
        'Monday': 'Lundi',
        'Tuesday': 'Mardi',
        'Wednesday': 'Mercredi',
        'Thursday': 'Jeudi',
        'Friday': 'Vendredi',
        'Saturday': 'Samedi',
        'Sunday': 'Dimanche'
    }

    for equip in equipments:
        df = report_data.get(equip)

        if df is None or df.empty:
            overview_rows.append({
                "Équipement": equip,
                "Disponibilité (%)": 0.0,
                "Durée Totale": "0 minute",
                "Périodes d'indisponibilité": 0,
                "Durée indisponible": format_minutes(0),
                "Périodes de données manquantes": 0,
                "Durée manquante": format_minutes(0)
            })

            equipment_details[equip] = EquipmentReportDetail(
                name=equip,
                summary=None,
                unavailable_table=pd.DataFrame(columns=["ID", "Date", "Jour", "Début", "Fin", "Durée", "Cause", "Exclu"]),
                missing_table=pd.DataFrame(columns=["ID", "Date", "Début", "Fin", "Durée", "Exclu"]),
                causes_table=pd.DataFrame(columns=["Cause", "Occurrences", "Durée (min)", "Durée Totale"]),
                daily_table=pd.DataFrame(columns=["Date", "Jour", "Nb Périodes", "Durée Totale", "Première Heure", "Dernière Heure", "% Journée"])
            )
            continue

        summary_row = df[df["ID"] == "RÉSUMÉ"].copy()
        detail_rows = df[df["ID"] != "RÉSUMÉ"].copy()

        summary_dict: Optional[Dict[str, str]] = None
        availability_pct = 0.0

        if not summary_row.empty:
            summary = summary_row.iloc[0]
            pct_match = re.search(r"(\d+\.?\d*)%", str(summary["Statut"]))
            availability_pct = float(pct_match.group(1)) if pct_match else 0.0
            availability_values.append(availability_pct)

            summary_dict = {
                "Disponibilité": str(summary["Statut"]),
                "Durée": str(summary["Durée"]),
                "Site": str(summary["Site"]),
                "Périodes": str(len(detail_rows))
            }

        if "Durée_Minutes" in detail_rows.columns:
            detail_rows["Durée_Minutes"] = detail_rows["Durée_Minutes"].fillna(0).astype(int)
        else:
            detail_rows["Durée_Minutes"] = 0

        unavailable = detail_rows[detail_rows["ID"].str.startswith("IND-")].copy()
        missing = detail_rows[detail_rows["ID"].str.startswith("MISS-")].copy()

        unavailable_minutes = int(unavailable["Durée_Minutes"].sum()) if not unavailable.empty else 0
        missing_minutes = int(missing["Durée_Minutes"].sum()) if not missing.empty else 0
        excluded_events = int(
            (unavailable.get("Exclu", pd.Series(dtype=str)) == "✅ Oui").sum() +
            (missing.get("Exclu", pd.Series(dtype=str)) == "✅ Oui").sum()
        )

        overview_rows.append({
            "Équipement": equip,
            "Disponibilité (%)": round(availability_pct, 2),
            "Durée Totale": summary_dict["Durée"] if summary_dict else "0 minute",
            "Périodes d'indisponibilité": len(unavailable),
            "Durée indisponible": format_minutes(unavailable_minutes),
            "Périodes de données manquantes": len(missing),
            "Durée manquante": format_minutes(missing_minutes)
        })

        def _with_dates(df_source: pd.DataFrame) -> pd.DataFrame:
            if df_source.empty:
                return df_source
            df_display = df_source.copy()
            df_display["Date"] = pd.to_datetime(df_display["Début"]).dt.strftime("%Y-%m-%d")
            df_display["Jour"] = pd.to_datetime(df_display["Début"]).dt.day_name().map(jours_fr)
            return df_display

        unavailable_display = _with_dates(unavailable)
        if not unavailable_display.empty:
            unavailable_display = unavailable_display.sort_values("Durée_Minutes", ascending=False)
            unavailable_display = unavailable_display[[
                "ID", "Date", "Jour", "Début", "Fin", "Durée", "Cause Traduite", "Exclu"
            ]].rename(columns={"Cause Traduite": "Cause"})

        missing_display = _with_dates(missing)
        if not missing_display.empty:
            missing_display = missing_display.sort_values("Durée_Minutes", ascending=False)
            missing_display = missing_display[["ID", "Date", "Début", "Fin", "Durée", "Exclu"]]

        if not unavailable.empty:
            causes_table = (
                unavailable.groupby("Cause Traduite", dropna=False)
                .agg(Occurrences=("ID", "count"), Durée_Minutes=("Durée_Minutes", "sum"))
                .reset_index()
                .sort_values(["Occurrences", "Durée_Minutes"], ascending=[False, False])
            )
            causes_table["Durée Totale"] = causes_table["Durée_Minutes"].apply(lambda x: format_minutes(int(x)))
            causes_table = causes_table.rename(columns={"Cause Traduite": "Cause", "Durée_Minutes": "Durée (min)"})
            causes_table = causes_table[["Cause", "Occurrences", "Durée (min)", "Durée Totale"]].head(5)
        else:
            causes_table = pd.DataFrame(columns=["Cause", "Occurrences", "Durée (min)", "Durée Totale"])

        if not unavailable.empty:
            daily_input = unavailable.rename(columns={"Début": "date_debut", "Fin": "date_fin"})
            daily_table = analyze_daily_unavailability(daily_input)
        else:
            daily_table = pd.DataFrame(columns=["Date", "Jour", "Nb Périodes", "Durée Totale", "Première Heure", "Dernière Heure", "% Journée"])

        equipment_details[equip] = EquipmentReportDetail(
            name=equip,
            summary=summary_dict,
            unavailable_table=unavailable_display,
            missing_table=missing_display,
            causes_table=causes_table,
            daily_table=daily_table,
            unavailable_minutes=unavailable_minutes,
            missing_minutes=missing_minutes,
            excluded_events=excluded_events
        )

        total_unavailable_minutes += unavailable_minutes
        total_missing_minutes += missing_minutes
        total_unavailable_events += len(unavailable)
        total_missing_events += len(missing)
        total_exclusions += excluded_events

    overview_df = pd.DataFrame(overview_rows)

    totals = {
        "average_availability": round(sum(availability_values) / len(availability_values), 2) if availability_values else 0.0,
        "unavailable_events": total_unavailable_events,
        "unavailable_minutes": total_unavailable_minutes,
        "missing_events": total_missing_events,
        "missing_minutes": total_missing_minutes,
        "excluded_events": total_exclusions
    }

    return overview_df, equipment_details, totals


def _render_equipment_detail(detail: EquipmentReportDetail) -> None:
    """Affiche la section détaillée d'un équipement."""

    icons = {"AC": "⚡", "DC1": "🔋", "DC2": "🔋"}
    st.markdown(f"#### {icons.get(detail.name, '🔧')} Équipement {detail.name}")

    if not detail.summary:
        st.info("ℹ️ Aucune donnée disponible pour cet équipement sur la période sélectionnée.")
        return

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Disponibilité", detail.summary.get("Disponibilité", "N/A"))
    with col2:
        st.metric("Durée analysée", detail.summary.get("Durée", "N/A"))
    with col3:
        st.metric("Site", detail.summary.get("Site", "N/A"))
    with col4:
        st.metric(
            "Périodes", detail.summary.get("Périodes", "0"),
            help=f"Indisponibilités: {format_minutes(detail.unavailable_minutes)} | Données manquantes: {format_minutes(detail.missing_minutes)}"
        )

    if not detail.unavailable_table.empty:
        with st.expander("Périodes d'indisponibilité", expanded=False):
            st.dataframe(
                detail.unavailable_table,
                hide_index=True,
                use_container_width=True,
                column_config={
                    "ID": st.column_config.TextColumn("ID", width="small"),
                    "Date": st.column_config.TextColumn("Date", width="small"),
                    "Jour": st.column_config.TextColumn("Jour", width="small"),
                    "Début": st.column_config.TextColumn("Début", width="medium"),
                    "Fin": st.column_config.TextColumn("Fin", width="medium"),
                    "Durée": st.column_config.TextColumn("Durée", width="medium"),
                    "Cause": st.column_config.TextColumn("Cause", width="large"),
                    "Exclu": st.column_config.TextColumn("Exclu", width="small")
                }
            )
    else:
        st.success("✅ Aucune indisponibilité détectée sur cette période.")

    if not detail.missing_table.empty:
        with st.expander("Périodes de données manquantes", expanded=False):
            st.dataframe(
                detail.missing_table,
                hide_index=True,
                use_container_width=True,
                column_config={
                    "ID": st.column_config.TextColumn("ID", width="small"),
                    "Date": st.column_config.TextColumn("Date", width="small"),
                    "Début": st.column_config.TextColumn("Début", width="medium"),
                    "Fin": st.column_config.TextColumn("Fin", width="medium"),
                    "Durée": st.column_config.TextColumn("Durée", width="medium"),
                    "Exclu": st.column_config.TextColumn("Exclu", width="small")
                }
            )

    if not detail.causes_table.empty:
        with st.expander("Top causes d'indisponibilité", expanded=False):
            st.dataframe(
                detail.causes_table,
                hide_index=True,
                use_container_width=True,
                column_config={
                    "Cause": st.column_config.TextColumn("Cause", width="large"),
                    "Occurrences": st.column_config.NumberColumn("Occurrences", width="small"),
                    "Durée (min)": st.column_config.NumberColumn("Durée (min)", width="small"),
                    "Durée Totale": st.column_config.TextColumn("Durée Totale", width="medium")
                }
            )

    if not detail.daily_table.empty:
        with st.expander("Répartition quotidienne", expanded=False):
            daily_sorted = detail.daily_table.copy()
            if "Durée_Minutes" in daily_sorted.columns:
                daily_sorted = daily_sorted.sort_values("Durée_Minutes", ascending=False)
            st.dataframe(
                daily_sorted[["Date", "Jour", "Nb Périodes", "Durée Totale", "Première Heure", "Dernière Heure", "% Journée"]],
                hide_index=True,
                use_container_width=True,
                column_config={
                    "Date": st.column_config.TextColumn("Date", width="small"),
                    "Jour": st.column_config.TextColumn("Jour", width="small"),
                    "Nb Périodes": st.column_config.NumberColumn("Nb Périodes", width="small"),
                    "Durée Totale": st.column_config.TextColumn("Durée Totale", width="medium"),
                    "Première Heure": st.column_config.TextColumn("Première Heure", width="small"),
                    "Dernière Heure": st.column_config.TextColumn("Dernière Heure", width="small"),
                    "% Journée": st.column_config.TextColumn("% Journée", width="small")
                }
            )



def render_report_tab():
    """Affiche l'onglet rapport de disponibilité."""
    mode = get_current_mode()
    st.header("📊 Rapport Exécutif de Disponibilité")

    if mode == MODE_PDC:
        st.markdown("""
        **Rapport complet** pour présentation et analyse des performances des points de charge.
        Cette vue regroupe toutes les métriques clés, analyses détaillées et recommandations spécifiques aux PDC.
        """)
    else:
        st.markdown("""
        **Rapport complet** pour présentation et analyse des performances des équipements AC, DC1, DC2.
        Cette vue regroupe toutes les métriques clés, analyses détaillées et recommandations.
        """)

    site_current = st.session_state.get("current_site")
    start_dt_current = st.session_state.get("current_start_dt")
    end_dt_current = st.session_state.get("current_end_dt")

    if not site_current:
        st.warning("⚠️ Sélectionnez un site spécifique pour générer le rapport.")
        return

    if not start_dt_current or not end_dt_current:
        st.warning("⚠️ Veuillez sélectionner une période dans les filtres pour générer le rapport.")
        return

    with st.spinner("⏳ Génération du rapport exécutif..."):
        report_data = generate_availability_report(start_dt_current, end_dt_current, site_current, mode=mode)

    if not report_data:
        st.warning("⚠️ Aucune donnée disponible pour générer le rapport.")
        return

    equipments = sorted(report_data.keys())
    if not equipments:
        equipments = get_equipments(mode, site_current)
    overview_df, equipment_details, totals = _prepare_report_summary(report_data, equipments)

    analysis_duration = end_dt_current - start_dt_current
    analysis_minutes = int(analysis_duration.total_seconds() // 60)
    if site_current:
        site_suffix = site_current.split("_")[-1]
        site_name = mapping_sites.get(site_suffix)
        site_label = (
            f"{site_current} – {site_name}"
            if site_name
            else site_current
        )
    else:
        site_label = "Tous les sites"
    equipments_available = sum(1 for detail in equipment_details.values() if detail.summary)

    st.markdown("---")

    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        st.markdown(f"""
        ### 📋 Informations du Rapport
        - **Site** : {site_label}
        - **Période analysée** : {start_dt_current.strftime('%d/%m/%Y')} → {end_dt_current.strftime('%d/%m/%Y')}
        - **Durée d'analyse** : {analysis_duration.days} jours
        - **Équipements analysés** : {equipments_available}
        """)
    with col2:
        st.metric("Date de génération", datetime.now().strftime("%d/%m/%Y"))
    with col3:
        st.metric("Heure de génération", datetime.now().strftime("%H:%M"))

    st.markdown("---")
    st.subheader("📊 Résumé Exécutif")

    metrics_cols = st.columns(4)
    with metrics_cols[0]:
        st.metric(
            "Disponibilité moyenne",
            f"{totals['average_availability']:.2f}%",
            help="Moyenne des disponibilités par équipement"
        )
    with metrics_cols[1]:
        st.metric(
            "Indisponibilités",
            totals["unavailable_events"],
            help=f"Durée cumulée: {format_minutes(totals['unavailable_minutes'])}"
        )
    with metrics_cols[2]:
        st.metric(
            "Données manquantes",
            totals["missing_events"],
            help=f"Durée cumulée: {format_minutes(totals['missing_minutes'])}"
        )
    with metrics_cols[3]:
        st.metric(
            "Périodes exclues",
            totals["excluded_events"],
            help="Nombre total d'intervalles exclus du calcul"
        )

    st.caption(f"Durée totale analysée : {format_minutes(analysis_minutes)}")

    st.markdown("**📈 Vue d'ensemble des équipements :**")
    if not overview_df.empty:
        overview_display = overview_df.copy()
        overview_display["Disponibilité (%)"] = overview_display["Disponibilité (%)"].map(lambda x: f"{x:.2f}%")
        st.dataframe(
            overview_display,
            hide_index=True,
            use_container_width=True,
            column_config={
                "Équipement": st.column_config.TextColumn("Équipement", width="small"),
                "Disponibilité (%)": st.column_config.TextColumn("Disponibilité (%)", width="medium"),
                "Durée Totale": st.column_config.TextColumn("Durée Totale", width="medium"),
                "Périodes d'indisponibilité": st.column_config.NumberColumn("Indisponibilités", width="small"),
                "Durée indisponible": st.column_config.TextColumn("Durée indisponible", width="medium"),
                "Périodes de données manquantes": st.column_config.NumberColumn("Données manquantes", width="small"),
                "Durée manquante": st.column_config.TextColumn("Durée manquante", width="medium")
            }
        )
    else:
        st.info("ℹ️ Aucune donnée disponible pour la période sélectionnée.")

    st.markdown("---")
    st.subheader("🔧 Analyse détaillée par équipement")

    for equip in equipments:
        detail = equipment_details.get(equip)
        if detail is None:
            st.info(f"ℹ️ Aucune donnée disponible pour {equip}.")
            continue
        _render_equipment_detail(detail)

    st.markdown("---")
    st.subheader("🛠️ Causes principales à analyser")

    all_causes: List[Dict[str, object]] = []
    for detail in equipment_details.values():
        if detail.causes_table.empty:
            continue
        for _, row in detail.causes_table.iterrows():
            all_causes.append({
                "equipement": detail.name,
                "cause": row["Cause"],
                "occurrences": int(row["Occurrences"]),
                "duree_min": int(row.get("Durée (min)", 0))
            })

    if all_causes:
        causes_df = pd.DataFrame(all_causes)
        causes_summary = (
            causes_df.groupby("cause", dropna=False)
            .agg(occurrences=("occurrences", "sum"), duree_min=("duree_min", "sum"))
            .reset_index()
            .sort_values(["occurrences", "duree_min"], ascending=[False, False])
        )
        top_causes = causes_summary.head(3)

        st.markdown("**🔍 Top 3 des causes principales :**")
        cols = st.columns(len(top_causes)) if len(top_causes) > 0 else []
        for idx, (_, cause_row) in enumerate(top_causes.iterrows()):
            with cols[idx]:
                st.metric(
                    f"Cause #{idx + 1}",
                    f"{int(cause_row['occurrences'])} occurrences",
                    help=f"Durée cumulée: {format_minutes(int(cause_row['duree_min']))}"
                )
        if not top_causes.empty:
            st.markdown("**📌 Points d'attention :**")
            for idx, cause_row in enumerate(top_causes.itertuples(), 1):
                st.markdown(
                    f"{idx}. **{cause_row.cause}** — {int(cause_row.occurrences)} occurrences, "
                    f"{format_minutes(int(cause_row.duree_min))} d'indisponibilité cumulée."
                )
    else:
        st.success("✅ Aucune indisponibilité détectée sur la période analysée. Excellente performance !")

CONTRACT_MONTHLY_TABLE = "dispo_contract_monthly"


def _month_bounds(start_dt: datetime, end_dt: datetime) -> Tuple[pd.Timestamp, pd.Timestamp]:
    start = pd.Timestamp(start_dt).to_period("M").to_timestamp()
    end = pd.Timestamp(end_dt).to_period("M").to_timestamp()
    return start, (end + pd.offsets.MonthBegin(1))


def load_stored_contract_monthly(
    site: str,
    start_dt: datetime,
    end_dt: datetime,
) -> pd.DataFrame:
    start_month, end_month = _month_bounds(start_dt, end_dt)
    query = f"""
        SELECT
            period_start,
            t2,
            t3,
            t_sum,
            availability_pct,
            notes,
            computed_at
        FROM {CONTRACT_MONTHLY_TABLE}
        WHERE site = :site
          AND period_start >= :start_month
          AND period_start < :end_month
        ORDER BY period_start
    """
    try:
        df = execute_query(
            query,
            {
                "site": site,
                "start_month": start_month.to_pydatetime(),
                "end_month": end_month.to_pydatetime(),
            },
        )
    except DatabaseError:
        return pd.DataFrame()

    if df.empty:
        return df

    df["period_start"] = pd.to_datetime(df["period_start"], errors="coerce")
    df["Mois"] = df["period_start"].dt.strftime("%Y-%m")
    df["T2"] = df["t2"].astype(int)
    df["T3"] = df["t3"].astype(int)
    df["T(11..16)"] = df["t_sum"].astype(float).round(2)
    df["Disponibilité (%)"] = df["availability_pct"].astype(float).round(2)
    df["Notes"] = df["notes"].fillna("")
    df["Calculé le"] = pd.to_datetime(df["computed_at"], errors="coerce")
    columns = [
        "Mois",
        "T2",
        "T3",
        "T(11..16)",
        "Disponibilité (%)",
        "Notes",
        "Calculé le",
    ]
    return df[columns].sort_values("Mois").reset_index(drop=True)


def render_contract_tab(site: Optional[str], start_dt: datetime, end_dt: datetime) -> None:
    """Affiche les règles contractuelles et charge la disponibilité mensuelle stockée."""
    st.header("📄 Disponibilité contractuel")

    st.markdown("### Formule générale")
    st.markdown(
        r"**Disponibilité (%)** = $\dfrac{T(11..16) + T_3}{T_2} \times 100$"
    )

    st.caption(
        "Le calcul s'effectue sur des pas de 10 minutes, obtenus en moyennant les états échantillonnés"
        " toutes les 5 secondes."
    )

    st.markdown("### Définitions")
    st.markdown(
        "- **T2** : Nombre total de pas de 10 minutes sur la période d'observation (mois ou année).\n"
        "- **T3** : Nombre de pas de 10 minutes durant lesquels la station est arrêtée sur décision"
        " externe (propriétaire, autorité locale, gestionnaire de réseau, maintenance préventive).\n"
        "- **T(11..16)** : Somme des disponibilités calculées pour tous les pas hors T3, à partir des"
        " six points de charge (T11 à T16) avec un poids de 1/6 chacun."
    )

    st.markdown("### Règles par pas (hors T3)")

    st.subheader("A. Condition préalable AC + Batteries")
    st.markdown(
        "- Le pas est pris en compte uniquement si le réseau AC et les batteries DC1 et DC2 sont en"
        " fonctionnement normal ou partiel."
    )
    st.markdown("- **AC indisponible** : la station est indisponible sur le pas (disponibilité = 0).")
    st.markdown(
        "- **Batteries** :\n"
        "  - Une seule colonne indisponible (DC1 **ou** DC2) → la station reste disponible, le calcul"
        " peut continuer.\n"
        "  - Plus d'une colonne indisponible → station indisponible sur le pas (disponibilité = 0)."
    )

    st.subheader("B. Règle PDC (T11…T16)")
    st.markdown(
        "- **1 à 2 PDC indisponibles simultanément** : appliquer un prorata égal au nombre de PDC"
        " disponibles divisé par 6."
    )
    st.markdown(
        "- **3 à 6 PDC indisponibles** : la station est considérée indisponible sur le pas (valeur 0)."
    )

    st.markdown("### Exemple pour un pas")
    st.markdown(
        "Si un PDC est indisponible 1 minute sur 10 et les cinq autres sont disponibles :"
    )
    st.latex(r"T_{pas} = \frac{0.9 + 1 + 1 + 1 + 1 + 1}{6} = 0.9833 \Rightarrow 98.33\%")

    st.markdown("### Agrégation finale sur la période")
    st.markdown(
        "- **T(11..16)** : somme des disponibilités $T_{pas}$ pour tous les pas hors T3.\n"
        "- **T3** : nombre total de pas exclus.\n"
        "- **T2** : nombre total de pas analysés sur la période.\n"
        r"- **Disponibilité (%)** : $\dfrac{T(11..16) + T_3}{T_2} \times 100$."
    )

    st.markdown("---")
    st.subheader("📅 Disponibilité contractuelle mensuelle")
    if not site:
        st.warning("Sélectionnez un site dans les filtres pour calculer la disponibilité contractuelle.")
        return

    with st.spinner("Chargement des indicateurs contractuels..."):
        monthly_df = load_stored_contract_monthly(site, start_dt, end_dt)

    if monthly_df.empty:
        st.info(
            "Aucune donnée contractuelle stockée pour cette période. "
            "Exécutez le script `python Dispo/contract_metrics_job.py <site> <debut> <fin>` "
            "pour alimenter le tableau."
        )
        return

    warning_messages = {
        note.strip()
        for note in monthly_df.get("Notes", pd.Series(dtype=str)).dropna().tolist()
        if note and note.strip()
    }
    for warning in sorted(warning_messages):
        st.warning(warning)

    global_availability = monthly_df["Disponibilité (%)"].mean()
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Disponibilité moyenne", f"{global_availability:.2f}%")
    with col2:
        total_steps = int(monthly_df["T2"].sum())
        st.metric("Nombre total de pas (T2)", f"{total_steps}")

    st.dataframe(
        monthly_df.drop(columns=["Notes"], errors="ignore"),
        hide_index=True,
        use_container_width=True,
        column_config={
            "Mois": st.column_config.TextColumn("Mois", width="medium"),
            "T2": st.column_config.NumberColumn("T2", width="small"),
            "T3": st.column_config.NumberColumn("T3", width="small"),
            "T(11..16)": st.column_config.NumberColumn("T(11..16)", format="%.2f"),
            "Disponibilité (%)": st.column_config.NumberColumn("Disponibilité (%)", format="%.2f"),
            "Calculé le": st.column_config.DatetimeColumn("Calculé le", format="YYYY-MM-DD HH:mm"),
        },
    )

    if "Calculé le" in monthly_df.columns and not monthly_df["Calculé le"].isna().all():
        last_update = monthly_df["Calculé le"].max()
        if pd.notna(last_update):
            st.caption(
                f"Dernière mise à jour contractuelle : {last_update.strftime('%Y-%m-%d %H:%M')}"
            )
    evo_df = monthly_df.copy()
    evo_df = evo_df[pd.notna(evo_df["Disponibilité (%)"])]
    evo_df["__mois_dt"] = pd.to_datetime(evo_df["Mois"] + "-01", errors="coerce")
    evo_df = evo_df.sort_values("__mois_dt")
    evo_df = evo_df.set_index("Mois")
    st.bar_chart(evo_df["Disponibilité (%)"])

def calcul():
    st.header("Réseau AC")
    with st.expander("AC"):
        with st.expander("Conditions de disponibilité"):
            st.markdown("- **Condition** : SEQ01.OLI.A.PC1 = `0` ET SEQ01.OLI.A.IC1 = `0`")
        with st.expander("Conditions d'indisponibilité"):
            st.markdown("Autres valeurs de SEQ01.OLI.A.IC1 et SEQ01.OLI.A.PC1")
            st.markdown("-- La cause d'indisponibilité :")
            st.markdown("  - SEQ01.OLI.A.PC1")
            st.markdown("  - SEQ01.OLI.A.IC1")

    st.header("Batterie DC1")
    with st.expander("DC1"):
        with st.expander("Conditions de disponibilité"):
            st.markdown("- SEQ02.OLI.A.PC1 = `0` ET SEQ02.OLI.A.IC1 = `0`")
            st.markdown("-- OU")
            st.markdown("- SEQ02.OLI.A.PC1 = `4` ET SEQ02.OLI.A.IC1 = `8`")
        with st.expander("Conditions d'indisponibilité"):
            st.markdown("Autres valeurs de SEQ02.OLI.A.IC1 et SEQ02.OLI.A.PC1")
            st.markdown("-- La cause d'indisponibilité :")
            st.markdown("  - SEQ02.OLI.A.PC1")
            st.markdown("  - SEQ02.OLI.A.IC1")

    st.header("Batterie DC2")
    with st.expander("DC2"):
        with st.expander("Conditions de disponibilité"):
            st.markdown("- **Condition** : SEQ03.OLI.A.PC1 = `0` ET SEQ03.OLI.A.IC1 = `0`")
            st.markdown("-- OU")
            st.markdown("- **Condition** : SEQ03.OLI.A.PC1 = `4` ET SEQ03.OLI.A.IC1 = `8`")
        with st.expander("Conditions d'indisponibilité"):
            st.markdown("Autres valeurs de SEQ03.OLI.A.IC1 et SEQ03.OLI.A.PC1")
            st.markdown("-- La cause d'indisponibilité :")
            st.markdown("  - SEQ03.OLI.A.PC1")
            st.markdown("  - SEQ03.OLI.A.IC1")
    st.header("Bornes PDC")

    def pdc_block(name, seq):
        with st.expander(name):
            with st.expander("Conditions de disponibilité"):
                st.markdown("- **Condition 1** : SEQ%s.OLI.A.IC1 = `1024`" % seq)
                st.markdown("- **Condition 2** : SEQ%s.OLI.A.IC1 = `0` ET SEQ%s.OLI.A.PC1 = `0`" % (seq, seq))
            with st.expander("Conditions d'indisponibilité"):
                st.markdown("Autres valeurs de SEQ%s.OLI.A.IC1 et SEQ%s.OLI.A.PC1" % (seq, seq))
                st.markdown("-- La cause d'indisponibilité :")
                st.markdown("  - SEQ%s.OLI.A.PC1" % seq)
                st.markdown("  - SEQ%s.OLI.A.IC1" % seq)
    pdc_block("PDC1", "12")
    pdc_block("PDC2", "22")
    pdc_block("PDC3", "13")
    pdc_block("PDC4", "23")
    pdc_block("PDC5", "14")
    pdc_block("PDC6", "24")


def render_statistics_tab() -> None:
    """Affiche la vue statistique multi-équipements pour chaque site."""

    st.header("📊 Vue Statistique Stations")
    st.caption("Analyse les indisponibilités critiques AC, DC et PDC en excluant les pertes de données.")

    available_sites = get_sites(MODE_EQUIPMENT)
    if not available_sites:
        st.warning("Aucun site disponible pour l'analyse statistique.")
        return

    current_site = st.session_state.get("current_site")
    if current_site and current_site in available_sites:
        default_sites = [current_site]
    else:
        default_sites = available_sites[:1]

    selected_sites = st.multiselect(
        "Sites à analyser",
        options=available_sites,
        default=default_sites,
        format_func=lambda code: mapping_sites.get(code.split("_")[-1], code),
        help="Sélectionnez un ou plusieurs sites pour visualiser leurs statistiques détaillées."
    )

    session_start = st.session_state.get("current_start_dt")
    session_end = st.session_state.get("current_end_dt")

    if not isinstance(session_start, datetime):
        session_start = datetime.now() - timedelta(days=7)
    if not isinstance(session_end, datetime):
        session_end = datetime.now()

    col_start, col_end = st.columns(2)
    start_date = col_start.date_input(
        "Date de début",
        value=session_start.date(),
        max_value=session_end.date(),
        help="Date de début de la fenêtre d'analyse statistique."
    )
    end_date = col_end.date_input(
        "Date de fin",
        value=session_end.date(),
        min_value=start_date,
        help="Date de fin de la fenêtre d'analyse statistique."
    )

    start_dt = datetime.combine(start_date, time.min)
    end_dt = datetime.combine(end_date, time.max)

    st.caption("Les métriques calculées considèrent la station indisponible dès qu'une condition critique est vraie.")

    if not selected_sites:
        st.info("Sélectionnez au moins un site pour afficher la vue statistique.")
        return

    for idx, site in enumerate(selected_sites, start=1):
        site_label = mapping_sites.get(site.split("_")[-1], site)
        st.subheader(f"📍 {site_label} ({site})")

        try:
            with st.spinner(f"Analyse des conditions critiques pour {site_label}..."):
                stats = load_station_statistics(site, start_dt, end_dt)
        except Exception as exc:
            logger.error("Erreur lors de l'analyse statistique pour %s : %s", site, exc)
            st.error(f"❌ Impossible de calculer les statistiques pour {site_label}. {exc}")
            if idx < len(selected_sites):
                st.divider()
            continue

        summary_df = stats.get("summary_df", pd.DataFrame())
        metrics = stats.get("metrics", {})
        timeline_df = stats.get("timeline_df", pd.DataFrame())
        condition_intervals = stats.get("condition_intervals", {})
        downtime_intervals = stats.get("downtime_intervals", [])

        availability_pct = float(metrics.get("availability_pct", 0.0) or 0.0)
        downtime_minutes = int(metrics.get("downtime_minutes", 0) or 0)
        reference_minutes = int(metrics.get("reference_minutes", 0) or 0)
        uptime_minutes = int(metrics.get("uptime_minutes", max(reference_minutes - downtime_minutes, 0)))
        window_minutes = int(metrics.get("window_minutes", 0) or 0)
        coverage_pct = float(metrics.get("coverage_pct", 0.0) or 0.0)
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Disponibilité estimée", f"{availability_pct:.2f}%")
        with col2:
            st.metric("Indisponibilité réelle de la station", format_minutes(downtime_minutes))
        with col3:
            st.metric(
                "Temps analysé",
                format_minutes(reference_minutes),
                help=f"{coverage_pct:.1f}% du total ({format_minutes(window_minutes)})"
            )

        if window_minutes > 0 and coverage_pct < 80:
            st.warning("Couverture partielle des données : certaines périodes n'ont pas pu être analysées.")

        if not summary_df.empty:
            display_df = summary_df.copy()
            display_df["Temps analysé"] = display_df["Temps_Analysé_Minutes"].apply(
                lambda m: format_minutes(int(m))
            )
            display_df["Durée"] = display_df["Durée_Minutes"].apply(
                lambda m: format_minutes(int(m))
            )

            ordered_columns = [
                "Condition",
                "Durée",
                "Temps analysé",
            ]

            display_df = display_df[ordered_columns]

            st.dataframe(
                display_df,
                hide_index=True,
                use_container_width=True,
                column_config={
                    "Condition": st.column_config.TextColumn("Condition", width="large"),
                    "Durée": st.column_config.TextColumn("Durée", width="medium"),
                    "Temps analysé": st.column_config.TextColumn("Temps analysé", width="medium"),
                }
            )
        else:
            st.success("Aucune condition critique détectée sur la période analysée.")

        for label, intervals in condition_intervals.items():
            interval_df = _build_interval_table(intervals)
            if interval_df.empty:
                continue
            with st.expander(f"Détails — {label} ({len(intervals)} période{'s' if len(intervals) > 1 else ''})"):
                table_display = interval_df.copy()
                table_display["Début"] = table_display["Début"].dt.strftime("%Y-%m-%d %H:%M")
                table_display["Fin"] = table_display["Fin"].dt.strftime("%Y-%m-%d %H:%M")
                table_display["Durée"] = table_display["Durée_Minutes"].apply(lambda m: format_minutes(int(m)))
                table_display = table_display.rename(columns={"Durée_Minutes": "Durée (min)"})
                st.dataframe(
                    table_display[["Période", "Début", "Fin", "Durée (min)", "Durée"]],
                    hide_index=True,
                    use_container_width=True,
                )

        downtime_df = _build_interval_table(downtime_intervals)
        if not downtime_df.empty:
            with st.expander(f"Périodes d'indisponibilité réelle de la station ({len(downtime_intervals)})"):
                dt_display = downtime_df.copy()
                dt_display["Début"] = dt_display["Début"].dt.strftime("%Y-%m-%d %H:%M")
                dt_display["Fin"] = dt_display["Fin"].dt.strftime("%Y-%m-%d %H:%M")
                dt_display["Durée"] = dt_display["Durée_Minutes"].apply(lambda m: format_minutes(int(m)))
                dt_display = dt_display.rename(columns={"Durée_Minutes": "Durée (min)"})
                st.dataframe(
                    dt_display[["Période", "Début", "Fin", "Durée (min)", "Durée"]],
                    hide_index=True,
                    use_container_width=True,
                )
        else:
            st.info("Aucune période d'indisponibilité réelle détectée pour la station.")

        if not timeline_df.empty:
            order = ["AC", "DC1", "DC2"] + [f"PDC{i}" for i in range(1, 7)]
            available_order = [item for item in order if item in timeline_df["Equipement"].unique()]
            if not available_order:
                available_order = timeline_df["Equipement"].unique().tolist()

            color_map = {
                "✅ Disponible": "#28a745",
                "❌ Indisponible": "#dc3545",
                "❌ Indisponible (Exclu)": "#fd7e14",
                "⚠️ Donnée manquante": "#6c757d",
                "⚠️ Donnée manquante (Exclu)": "#BBDB07",
                "❓ Inconnu": "#adb5bd",
                "❓ Inconnu (Exclu)": "#868e96",
            }

            fig = px.timeline(
                timeline_df,
                x_start="start",
                x_end="end",
                y="Equipement",
                color="label",
                hover_data={
                    "cause": True,
                    "duration_minutes": True,
                    "start": "|%Y-%m-%d %H:%M",
                    "end": "|%Y-%m-%d %H:%M",
                    "Equipement": False,
                    "label": False,
                },
                category_orders={"Equipement": available_order},
                color_discrete_map=color_map,
            )
            fig.update_yaxes(autorange="reversed", title="")
            fig.update_xaxes(title="Période")
            base_height = 120 + 40 * len(available_order)
            fig.update_layout(
                height=max(360, base_height),
                showlegend=True,
                title=f"Timeline des équipements — {site_label}",
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Aucune donnée de timeline disponible pour cette période.")

        st.caption(f"Temps disponible estimé : {format_minutes(uptime_minutes)}")

        if idx < len(selected_sites):
            st.divider()


def main():
    """Point d'entrée principal de l'application."""
    
    if "last_cache_clear" not in st.session_state:
        st.session_state["last_cache_clear"] = None
    
    render_header()
    
    st.divider()
    
    site, equip, start_dt, end_dt = render_filters()

    selection_valid = site is not None and equip is not None

    st.session_state["current_site"] = site if selection_valid else None
    st.session_state["current_equip"] = equip if selection_valid else None
    st.session_state["current_start_dt"] = start_dt
    st.session_state["current_end_dt"] = end_dt
    st.session_state["current_mode"] = get_current_mode()

    st.divider()

    if not selection_valid:
        st.error("⚠️ Sélectionnez un site et un équipement spécifiques pour afficher la disponibilité détaillée.")
        df_filtered = pd.DataFrame()
    else:
        with st.spinner("⏳ Chargement des données..."):
            df_filtered = load_filtered_blocks(start_dt, end_dt, site, equip, mode=get_current_mode())

    if df_filtered is None:
        logger.warning("Aucune donnée reçue de load_filtered_blocks, utilisation d'un DataFrame vide")
        df_filtered = pd.DataFrame()

    if not df_filtered.empty:
        st.caption(f"📊 {len(df_filtered)} blocs chargés pour la période sélectionnée")
    
    tabs = st.tabs([
        "📈 Vue d'ensemble",
        "📊 Vue statistique",
        "🌍 Comparaison sites",
        "⏱️ Timeline & Annotations",
        "📊 Rapport",
        "🚫 Exclusions",
        "💬 Commentaires",
        "ℹ️ Info calcul",
        "📄 Contrat",
    ])

    with tabs[0]:
        render_overview_tab(df_filtered)

    with tabs[1]:
        render_statistics_tab()

    with tabs[2]:
        render_global_comparison_tab(start_dt, end_dt)

    with tabs[3]:
        render_timeline_tab(site, equip, start_dt, end_dt)

    with tabs[4]:
        render_report_tab()

    with tabs[5]:
        render_exclusions_tab()

    with tabs[6]:
        render_comments_tab()

    with tabs[7]:
        calcul()

    with tabs[8]:
        render_contract_tab(site, start_dt, end_dt)

    st.divider()
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.caption("🔧 Dashboard Disponibilité v6.0")
    
    with col2:
        if st.session_state.get("last_cache_clear"):
            last_update = pd.to_datetime(st.session_state["last_cache_clear"]).strftime("%H:%M:%S")
            st.caption(f"🔄 Dernier rafraîchissement: {last_update}")
    
    with col3:
        st.caption("📞 Support: Nidec-ASI")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.exception("Erreur fatale dans l'application")
        st.error(f"""
        ❌ **Erreur Critique**
        
        Une erreur inattendue s'est produite:
        ```
        {str(e)}
        ```
        
        Veuillez contacter le support technique si le problème persiste.
        """)
        
        if st.button("🔄 Redémarrer l'application"):
            st.rerun()
