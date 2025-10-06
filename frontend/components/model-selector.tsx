"use client"

import { Button } from "@/components/ui/button"
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuLabel,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu"
import { Badge } from "@/components/ui/badge"
import { ChevronDown, Code, MessageCircle } from "lucide-react"
import type { LLMModel } from "@/app/page"

interface ModelSelectorProps {
  models: LLMModel[]
  selectedModel: LLMModel
  onModelChange: (model: LLMModel) => void
}

export function ModelSelector({ models, selectedModel, onModelChange }: ModelSelectorProps) {
  const codingModels = models.filter((m) => m.category === "coding")
  const conversationModels = models.filter((m) => m.category === "conversation")

  return (
    <div className="flex items-center gap-2">
      {codingModels.map((model) => (
        <Button
          key={model.id}
          variant={selectedModel.id === model.id ? "default" : "outline"}
          onClick={() => onModelChange(model)}
          className="gap-2 bg-transparent"
        >
          <Code className="w-4 h-4" />
          {model.name}
        </Button>
      ))}
      {conversationModels.map((model) => (
        <Button
          key={model.id}
          variant={selectedModel.id === model.id ? "default" : "outline"}
          onClick={() => onModelChange(model)}
          className="gap-2 bg-transparent"
        >
          <MessageCircle className="w-4 h-4" />
          {model.name}
        </Button>
      ))}
    </div>
  )
}
