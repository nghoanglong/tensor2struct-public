import os
import attr
import collections
import itertools
import timeout_decorator


from tensor2struct.contexts import abstract_context
from tensor2struct.utils import registry, serialization
from tensor2struct.contexts.spider_context import SpiderContext
from tensor2struct.models.spider.spider_enc_bert import PhoBertokens

@attr.s
class PreprocessedSchema:
    column_names = attr.ib(factory=list)
    table_names = attr.ib(factory=list)
    table_bounds = attr.ib(factory=list)
    column_to_table = attr.ib(factory=dict)
    table_to_columns = attr.ib(factory=dict)
    foreign_keys = attr.ib(factory=dict)
    foreign_keys_tables = attr.ib(factory=lambda: collections.defaultdict(set))
    primary_keys = attr.ib(factory=list)

    normalized_column_names = attr.ib(factory=list)
    normalized_table_names = attr.ib(factory=list)


@registry.register("context", "vitext2sql-phobert")
class Vitext2SQLPhoBertContext(SpiderContext):
    def __init__(self, schema, tokenizer, db_paths) -> None:
        self.tokenizer = tokenizer

        self.schema = schema
        preproc_schema = self.preprocess_schema(self.schema)
        self.preproc_schema = preproc_schema
        self.columns = [col[1:] for col in preproc_schema.column_names]
        self.tables = preproc_schema.table_names

        self.db_dir = db_paths

    def _normalize(self, sent_text, ret_meta=False):
        if not ret_meta:
            return PhoBertokens(sent_text)
        else:
            return PhoBertokens(sent_text)
    
    def preprocess_schema(self, schema):
        r = PreprocessedSchema()
        last_table_id = None
        for i, column in enumerate(schema.columns):

            # assert column.type in ["text", "number", "time", "boolean", "others"]
            # type_tok = "<type: {}>".format(column.type)
            # for bert, we take the representation of the first word

            col_toks = self.tokenizer(
                column.name, column.unsplit_name)
            r.normalized_column_names.append(self._normalize(col_toks))

            column_text = " ".join(col_toks)
            r.column_names.append(column_text)

            table_id = None if column.table is None else column.table.id
            r.column_to_table[str(i)] = table_id
            if table_id is not None:
                columns = r.table_to_columns.setdefault(str(table_id), [])
                columns.append(i)
            if last_table_id != table_id:
                r.table_bounds.append(i)
                last_table_id = table_id

            if column.foreign_key_for is not None:
                r.foreign_keys[str(column.id)] = column.foreign_key_for.id
                r.foreign_keys_tables[str(column.table.id)].add(
                    column.foreign_key_for.table.id
                )

        r.table_bounds.append(len(schema.columns))
        assert len(r.table_bounds) == len(schema.tables) + 1

        for i, table in enumerate(schema.tables):
            table_toks = self.tokenizer(
                table.name, table.unsplit_name)
            table_text = " ".join(table_toks)
            r.table_names.append(table_text)
            r.normalized_table_names.append(self._normalize(table_toks))

        r.foreign_keys_tables = serialization.to_dict_with_sorted_values(
            r.foreign_keys_tables
        )
        r.primary_keys = [
            column.id for table in schema.tables for column in table.primary_keys
        ]
        return r

