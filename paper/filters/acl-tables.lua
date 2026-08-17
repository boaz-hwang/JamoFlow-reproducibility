-- Render Pandoc pipe tables as ACL-compatible floats rather than longtable.

if not FORMAT:match("latex") then
  return
end

local function latex_blocks(blocks)
  local value = pandoc.write(pandoc.Pandoc(blocks), "latex")
  value = value:gsub("^%s+", "")
  value = value:gsub("%s+$", "")
  return value
end

local function latex_inlines(inlines)
  return latex_blocks({pandoc.Plain(inlines)})
end

local function cell_text(cell)
  return latex_blocks(cell.contents)
end

local function alignment_code(alignment)
  local name = tostring(alignment)
  if name:match("Right") then
    return "r"
  end
  if name:match("Center") then
    return "c"
  end
  return "l"
end

local function collect_rows(tbl)
  local rows = {}
  for _, body in ipairs(tbl.bodies) do
    for _, row in ipairs(body.head) do
      table.insert(rows, row)
    end
    for _, row in ipairs(body.body) do
      table.insert(rows, row)
    end
  end
  for _, row in ipairs(tbl.foot.rows) do
    table.insert(rows, row)
  end
  return rows
end

function Table(tbl)
  local column_count = #tbl.colspecs
  local wide = column_count >= 5
  local environment = wide and "table*" or "table"
  local target_width = wide and "\\textwidth" or "\\columnwidth"
  local alignments = {}
  for _, colspec in ipairs(tbl.colspecs) do
    table.insert(alignments, alignment_code(colspec[1]))
  end

  local lines = {
    "\\begin{" .. environment .. "}[t]",
    "\\centering",
    "\\small",
    "\\setlength{\\tabcolsep}{3pt}",
    "\\resizebox{" .. target_width .. "}{!}{%",
    "\\begin{tabular}{@{}" .. table.concat(alignments, "") .. "@{}}",
    "\\toprule",
  }

  if #tbl.head.rows > 0 then
    local header = {}
    for _, cell in ipairs(tbl.head.rows[1].cells) do
      table.insert(header, "\\textbf{" .. cell_text(cell) .. "}")
    end
    table.insert(lines, table.concat(header, " & ") .. " \\\\")
    table.insert(lines, "\\midrule")
  end

  for _, row in ipairs(collect_rows(tbl)) do
    local cells = {}
    for _, cell in ipairs(row.cells) do
      table.insert(cells, cell_text(cell))
    end
    table.insert(lines, table.concat(cells, " & ") .. " \\\\")
  end

  table.insert(lines, "\\bottomrule")
  table.insert(lines, "\\end{tabular}%")
  table.insert(lines, "}")

  if #tbl.caption.long > 0 then
    table.insert(lines, "\\caption{" .. latex_blocks(tbl.caption.long) .. "}")
  elseif #tbl.caption.short > 0 then
    table.insert(lines, "\\caption{" .. latex_inlines(tbl.caption.short) .. "}")
  end
  if tbl.identifier ~= "" then
    table.insert(lines, "\\label{" .. tbl.identifier .. "}")
  end
  table.insert(lines, "\\end{" .. environment .. "}")
  return pandoc.RawBlock("latex", table.concat(lines, "\n"))
end
