"use client"

import { useState } from "react"
import { Button } from "@/components/ui/button"
import { Copy, Check } from "lucide-react"

interface CodeBlockProps {
  content: string
}

export function CodeBlock({ content }: CodeBlockProps) {
  const [copied, setCopied] = useState(false)

  // Simple code detection - looks for common code patterns
  const isCode = (text: string) => {
    const codePatterns = [
      /```[\s\S]*```/, // Code blocks
      /`[^`]+`/, // Inline code
      /^\s*(?:function|class|def|import|from|const|let|var|if|for|while)\s/m, // Keywords
      /[{}();]/, // Common code symbols
      /^\s*\/\/|^\s*#|^\s*\/\*/m, // Comments
    ]
    return codePatterns.some((pattern) => pattern.test(text))
  }

  const copyToClipboard = async () => {
    try {
      await navigator.clipboard.writeText(content)
      setCopied(true)
      setTimeout(() => setCopied(false), 2000)
    } catch (err) {
      console.error("Failed to copy text: ", err)
    }
  }

  // Extract code blocks and format them
  const formatContent = (text: string) => {
    // Handle markdown code blocks
    const codeBlockRegex = /```(\w+)?\n?([\s\S]*?)```/g
    const parts = []
    let lastIndex = 0
    let match

    while ((match = codeBlockRegex.exec(text)) !== null) {
      // Add text before code block
      if (match.index > lastIndex) {
        const beforeText = text.slice(lastIndex, match.index)
        if (beforeText.trim()) {
          parts.push({
            type: "text",
            content: beforeText.trim(),
            key: `text-${lastIndex}`,
          })
        }
      }

      // Add code block
      parts.push({
        type: "code",
        language: match[1] || "text",
        content: match[2].trim(),
        key: `code-${match.index}`,
      })

      lastIndex = match.index + match[0].length
    }

    // Add remaining text
    if (lastIndex < text.length) {
      const remainingText = text.slice(lastIndex)
      if (remainingText.trim()) {
        parts.push({
          type: "text",
          content: remainingText.trim(),
          key: `text-${lastIndex}`,
        })
      }
    }

    // If no code blocks found, treat as single piece
    if (parts.length === 0) {
      parts.push({
        type: isCode(text) ? "code" : "text",
        content: text,
        language: "text",
        key: "single",
      })
    }

    return parts
  }

  const contentParts = formatContent(content)

  return (
    <div className="space-y-2">
      {contentParts.map((part) => {
        if (part.type === "code") {
          return (
            <div key={part.key} className="relative group">
              <Button
                variant="ghost"
                size="sm"
                className="absolute top-2 right-2 opacity-0 group-hover:opacity-100 transition-opacity z-10"
                onClick={copyToClipboard}
              >
                {copied ? <Check className="w-4 h-4" /> : <Copy className="w-4 h-4" />}
              </Button>
              <pre className="bg-muted p-4 rounded-lg overflow-x-auto text-sm font-mono">
                <code className={`language-${part.language}`}>{part.content}</code>
              </pre>
            </div>
          )
        } else {
          return (
            <div key={part.key} className="prose prose-sm max-w-none">
              <p className="whitespace-pre-wrap break-words">{part.content}</p>
            </div>
          )
        }
      })}
    </div>
  )
}
