---
title: "mcp>=2.0 handler が TypeError: takes 1 positional argument but 2 were given"
tags: [mcp, python, api]
severity: high
date: "2026-08-08"
---

## 症状

`server.add_request_handler("tools/list", ...)` を登録した handler を実行すると
`TypeError: handle_list_tools() takes 1 positional argument but 2 were given` が発生。
または `tools/call` で `-32602 INVALID_PARAMS` が返る。

## 原因

mcp>=2.0 のランタイムは `handler(ctx, typed_params)` の2引数で呼び出す。
`params_type` に `ListToolsRequest` などの full Request 型を渡すと、
`{"method","params"}` 構造に対してバリデーションが走り `-32602 INVALID_PARAMS` になる。

## 解決策

```python
# NG (1引数・full Request 型)
async def handle_list_tools(_req: types.ListToolsRequest):
server.add_request_handler("tools/list", types.ListToolsRequest, handle_list_tools)

# OK (2引数・Params サブモデル)
async def handle_list_tools(_ctx: object, _params: types.PaginatedRequestParams):
server.add_request_handler("tools/list", types.PaginatedRequestParams, handle_list_tools)

# tools/call も同様
async def handle_call_tool(_ctx: object, params: types.CallToolRequestParams):
    name = params.name  # params はサブモデル直参照
server.add_request_handler("tools/call", types.CallToolRequestParams, handle_call_tool)
```

## 予防

mcp パッケージのバージョンアップ後は必ず `runner.py` で呼び出し規約を確認する。
params_type は必ず Params サブモデル(メソッド名+Params suffix)を指定する。
