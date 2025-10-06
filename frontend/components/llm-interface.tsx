"use client"

import type React from "react"

import { useState } from "react"
import { Button } from "@/components/ui/button"
import { Textarea } from "@/components/ui/textarea"
import { Card, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Send, Loader2 } from "lucide-react"
import type { LLMModel } from "@/app/page"

interface LLMInterfaceProps {
  selectedModel: LLMModel
  onSendMessage: (prompt: string) => void
  isLoading: boolean
}

export function LLMInterface({ selectedModel, onSendMessage, isLoading }: LLMInterfaceProps) {
  const [prompt, setPrompt] = useState("")

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault()
    if (prompt.trim() && !isLoading) {
      onSendMessage(prompt)
      setPrompt("")
    }
  }

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) {
      e.preventDefault()
      handleSubmit(e)
    }
  }

  return (
    <div className="space-y-4">
      <Card className="border-primary/20">
        <CardHeader className="pb-3">
          <div className="flex items-center justify-between">
            <CardTitle className="text-lg">Selected Model</CardTitle>
            <Badge variant={selectedModel.category === "coding" ? "default" : "secondary"} className="capitalize">
              {selectedModel.category}
            </Badge>
          </div>
          <div className="flex items-center gap-2">
            <h3 className="font-semibold text-primary">{selectedModel.name}</h3>
            <span className="text-sm text-muted-foreground">•</span>
            <p className="text-sm text-muted-foreground">{selectedModel.description}</p>
          </div>
        </CardHeader>
      </Card>

      <form onSubmit={handleSubmit} className="space-y-4">
        <div className="space-y-2">
          <label htmlFor="prompt" className="text-sm font-medium">
            Enter your prompt
          </label>
          <Textarea
            id="prompt"
            placeholder={
              selectedModel.category === "coding"
                ? "Enter your coding question or request..."
                : "Enter your question or start a conversation..."
            }
            value={prompt}
            onChange={(e) => setPrompt(e.target.value)}
            onKeyDown={handleKeyDown}
            className="min-h-[120px] resize-none font-mono text-sm"
            disabled={isLoading}
          />
          <p className="text-xs text-muted-foreground">Press Ctrl+Enter (Cmd+Enter on Mac) to send</p>
        </div>

        <div className="flex gap-2">
          <Button type="submit" disabled={!prompt.trim() || isLoading} className="flex-1">
            {isLoading ? (
              <>
                <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                Generating...
              </>
            ) : (
              <>
                <Send className="w-4 h-4 mr-2" />
                Send Message
              </>
            )}
          </Button>
        </div>
      </form>
    </div>
  )
}
