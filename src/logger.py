import logging
from pathlib import Path

def setup_logger(logs_dir: str, run_id: str):
    Path(logs_dir).mkdir(parents=True, exist_ok=True)
    log_file = Path(logs_dir) / f"run_{run_id}.log"

    # створює іменований логер, назва "dp"
    logger = logging.getLogger("dp")
    logger.setLevel(logging.INFO)
    # видаляє старі налаштування, щоб при повторному запуску в одному середовищі 
    # кожне повідомлення дублювалося
    logger.handlers.clear()

    # визначає зовнішній вигляд рядка
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    # Записує всі події у файл на диску (.log)
    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setFormatter(fmt)
    # Виводить ті самі повідомлення прямо тобі в термінал
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)

    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger, str(log_file)
