from pathlib import Path

import pandas as pd
from langchain_core.documents import Document

from rag_chatbot.ingestion.base_loader import BaseDocumentLoader


class ExcelDocumentLoader(BaseDocumentLoader):
    """Load each populated Excel row as a separate document."""

    def load(self, file_path: Path) -> list[Document]:
        workbook = pd.ExcelFile(file_path)
        documents = []

        for sheet_name in workbook.sheet_names:
            dataframe = pd.read_excel(workbook, sheet_name=sheet_name)
            dataframe = dataframe.dropna(how="all")

            for row_number, row in enumerate(dataframe.iterrows(), start=2):
                _, values = row

                fields = [
                    f"{column}: {value}"
                    for column, value in values.items()
                    if pd.notna(value)
                ]

                if not fields:
                    continue

                documents.append(
                    Document(
                        page_content="\n".join(fields),
                        metadata={
                            "source": str(file_path),
                            "file_name": file_path.name,
                            "file_type": file_path.suffix.lower(),
                            "sheet": sheet_name,
                            "row": row_number,
                        },
                    )
                )

        return documents